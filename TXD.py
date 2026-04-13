import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseTextureDiffusion(nn.Module):
    """
    基础纹理扩散模块，作为条件扩散的基类
    """

    def __init__(self, latent_dim=24, window_size=7, max_steps=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.max_steps = max_steps

        # 基础纹理特征提取
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_diffusion_steps(self, height, width, num_iterations=None):
        """计算扩散迭代次数"""
        r = self.window_size
        if num_iterations is not None:
            return num_iterations
        if self.max_steps is not None:
            return self.max_steps
        return int(np.ceil(max(height, width) / (r // 2)))

    def _diffusion_step(self, latent, weights, window_size):
        """优化的单步扩散（使用unfold并行处理）"""
        batch_size, channels, height, width = latent.shape
        r = window_size
        pad = r // 2

        latent_padded = F.pad(latent, (pad, pad, pad, pad), mode='replicate')

        # 使用 unfold 提取所有窗口
        windows = F.unfold(latent_padded, kernel_size=r, padding=0)
        windows = windows.view(batch_size, channels, r * r, height, width)
        windows = windows.permute(0, 1, 3, 4, 2)  # [B, C, H, W, r*r]

        # 权重: [B, C, r*r, H, W] -> [B, C, H, W, r*r]
        weights_permuted = weights.permute(0, 1, 3, 4, 2)

        # 批量计算加权和
        updated = (windows * weights_permuted).sum(dim=-1)
        return updated


class AttentionGuidedTextureDiffusion(BaseTextureDiffusion):
    """
    注意力引导的条件纹理扩散

    使用注意力图在关键区域（如病灶边界）增强纹理扩散
    """

    def __init__(self, latent_dim=24, window_size=7, max_steps=None):
        super().__init__(latent_dim, window_size, max_steps)

        # 注意力编码器
        self.attention_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 条件融合模块 - 融合纹理特征和注意力特征
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(128 + 64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_dim * window_size * window_size, kernel_size=1)
        )

        # 空间门控模块 - 生成注意力权重图
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 不确定性估计器（可选）
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, depth_latent, texture_features, attention_map,
                uncertainty_map=None, num_iterations=None):
        """
        Args:
            depth_latent: 深度图潜在表示 [B, C, H, W]
            texture_features: 纹理特征 [B, 3, H, W]
            attention_map: 注意力图 [B, 1, H, W] (如CAM、Grad-CAM)
            uncertainty_map: 不确定性图 [B, 1, H, W] (可选)
            num_iterations: 迭代次数

        Returns:
            enhanced_latent: 增强后的潜在表示 [B, C, H, W]
            spatial_gate: 空间门控权重 [B, 1, H, W]
            attention_weights: 融合后的注意力权重
        """
        batch_size, channels, height, width = depth_latent.shape
        r = self.window_size

        # 1. 提取纹理特征
        tex_features = self.texture_encoder(texture_features)  # [B, 128, H, W]

        # 2. 编码注意力图
        att_features = self.attention_encoder(attention_map)  # [B, 64, H, W]

        # 3. 生成空间门控
        spatial_gate = self.spatial_gate(att_features)  # [B, 1, H, W]

        # 4. 如果有不确定性图，调整门控
        if uncertainty_map is not None:
            # 在高不确定性区域增强纹理扩散
            uncertainty_weight = self.uncertainty_estimator(tex_features)
            spatial_gate = spatial_gate * (1 + 0.5 * uncertainty_weight * uncertainty_map)

        # 5. 融合纹理特征和注意力特征
        fused_features = torch.cat([tex_features, att_features], dim=1)  # [B, 192, H, W]

        # 6. 预测扩散权重
        weights = self.fusion_layer(fused_features)
        weights = weights.view(batch_size, self.latent_dim, r * r, height, width)

        # 7. 应用空间门控调整权重
        spatial_gate_expanded = spatial_gate.view(batch_size, 1, 1, height, width)
        weights = weights * spatial_gate_expanded
        weights = F.softmax(weights, dim=2)  # [B, C, r*r, H, W]

        # 8. 确定迭代次数并执行扩散
        S = self._get_diffusion_steps(height, width, num_iterations)
        # print(f"注意力引导扩散 - 迭代次数: {S}, 窗口大小: {r}")

        enhanced = depth_latent.clone()
        for step in range(S):
            enhanced = self._diffusion_step(enhanced, weights, r)

        # 计算注意力权重（用于可视化）
        attention_weights = spatial_gate * attention_map

        return enhanced, spatial_gate, attention_weights


class CategoryConditionedTextureDiffusion(BaseTextureDiffusion):
    """
    类别条件纹理扩散

    根据病理图像的类别动态调整纹理扩散策略
    """

    def __init__(self, latent_dim=24, window_size=7, num_classes=6, max_steps=None):
        super().__init__(latent_dim, window_size, max_steps)
        self.num_classes = num_classes

        # 类别嵌入层
        self.class_embedding = nn.Embedding(num_classes, 128)

        # 条件融合模块 - 融合纹理特征和类别条件
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(128 + 128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_dim * window_size * window_size, kernel_size=1)
        )

        # 类别感知的门控模块
        self.class_gate = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

        # 类别特定的扩散强度（可学习参数）- 修改为 [1, num_classes, 1, 1] 以便广播
        self.class_intensity = nn.Parameter(torch.ones(1, num_classes, 1, 1))

    def forward(self, depth_latent, texture_features, class_labels,
                class_confidence=None, num_iterations=None):
        """
        Args:
            depth_latent: 深度图潜在表示 [B, C, H, W]
            texture_features: 纹理特征 [B, 3, H, W]
            class_labels: 类别标签，支持多种格式
                         - [B] (整数标签)
                         - [B, num_classes] (one-hot)
                         - [B, num_classes, H, W] (像素级类别)
            class_confidence: 类别置信度 [B, 1] 或 [B, num_classes] (可选)
            num_iterations: 迭代次数

        Returns:
            enhanced_latent: 增强后的潜在表示 [B, C, H, W]
            class_weights: 类别权重图 [B, num_classes, H, W]
            condition_map: 条件融合图 [B, 1, H, W]
        """
        batch_size, channels, height, width = depth_latent.shape
        r = self.window_size

        # 1. 提取纹理特征
        tex_features = self.texture_encoder(texture_features)  # [B, 128, H, W]

        # 2. 处理类别条件
        if class_labels.dim() == 1:
            # [B] -> [B, num_classes]
            class_onehot = F.one_hot(class_labels, num_classes=self.num_classes).float()
            class_onehot = class_onehot.view(batch_size, self.num_classes, 1, 1)
            class_cond = class_onehot.expand(-1, -1, height, width)

            # 类别嵌入
            class_emb = self.class_embedding(class_labels)  # [B, 128]
            class_emb = class_emb.view(batch_size, 128, 1, 1)
            class_emb = class_emb.expand(-1, -1, height, width)

        elif class_labels.dim() == 2 and class_labels.shape[1] == self.num_classes:
            # [B, num_classes] -> [B, num_classes, H, W]
            class_onehot = class_labels.view(batch_size, self.num_classes, 1, 1)
            class_cond = class_onehot.expand(-1, -1, height, width)

            # 类别嵌入（通过线性层）
            class_emb = torch.matmul(class_labels, self.class_embedding.weight)  # [B, 128]
            class_emb = class_emb.view(batch_size, 128, 1, 1)
            class_emb = class_emb.expand(-1, -1, height, width)

        elif class_labels.dim() == 4:
            # [B, num_classes, H, W] 像素级类别
            class_cond = class_labels
            # 对每个像素位置，取最大概率的类别进行嵌入
            class_indices = class_labels.argmax(dim=1)  # [B, H, W]
            class_emb = self.class_embedding(class_indices)  # [B, H, W, 128]
            class_emb = class_emb.permute(0, 3, 1, 2)  # [B, 128, H, W]
        else:
            raise ValueError(f"Unsupported class_labels shape: {class_labels.shape}")

        # 3. 生成类别门控
        class_gate = self.class_gate(tex_features)  # [B, num_classes, H, W]
        class_gate = class_gate * class_cond  # 根据实际类别过滤

        # 4. 应用类别置信度调整
        if class_confidence is not None:
            if class_confidence.dim() == 2:
                class_confidence = class_confidence.view(batch_size, -1, 1, 1)
                class_gate = class_gate * class_confidence

        # 5. 融合纹理特征和类别条件
        fused_features = torch.cat([tex_features, class_emb], dim=1)  # [B, 256, H, W]

        # 6. 预测扩散权重
        weights = self.fusion_layer(fused_features)
        weights = weights.view(batch_size, self.latent_dim, r * r, height, width)

        # 7. 应用类别特定的扩散强度 - 修复广播问题
        # class_intensity: [1, num_classes, 1, 1]
        # class_gate: [B, num_classes, H, W]
        # 两者相乘会自动广播到 [B, num_classes, H, W]
        class_intensity_map = (class_gate * self.class_intensity)  # [B, num_classes, H, W]
        class_intensity_map = class_intensity_map.sum(dim=1, keepdim=True)  # [B, 1, H, W]

        weights = weights * class_intensity_map.view(batch_size, 1, 1, height, width)
        weights = F.softmax(weights, dim=2)

        # 8. 确定迭代次数并执行扩散
        S = self._get_diffusion_steps(height, width, num_iterations)
        # print(f"类别条件扩散 - 迭代次数: {S}, 窗口大小: {r}")

        enhanced = depth_latent.clone()
        for step in range(S):
            enhanced = self._diffusion_step(enhanced, weights, r)

        # 计算条件融合图（用于可视化）
        condition_map = class_intensity_map * class_gate.sum(dim=1, keepdim=True)

        return enhanced, class_gate, condition_map


class HybridConditionedTextureDiffusion(BaseTextureDiffusion):
    """
    混合条件纹理扩散（注意力引导 + 类别条件）

    结合两种条件的优势，用于病理图像分割
    """

    def __init__(self, latent_dim=24, window_size=7, num_classes=6, max_steps=None):
        super().__init__(latent_dim, window_size, max_steps)

        # 注意力引导分支
        self.attention_branch = AttentionGuidedTextureDiffusion(
            latent_dim, window_size, max_steps
        )

        # 类别条件分支
        self.category_branch = CategoryConditionedTextureDiffusion(
            latent_dim, window_size, num_classes, max_steps
        )

        # 自适应融合权重
        self.fusion_weight_attention = nn.Parameter(torch.tensor(0.5))
        self.fusion_weight_category = nn.Parameter(torch.tensor(0.5))

        # 元网络：根据输入特征动态调整融合权重
        self.meta_fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, depth_latent, texture_features, attention_map,
                class_labels, uncertainty_map=None, class_confidence=None,
                num_iterations=None):
        """
        混合条件扩散前向传播

        Returns:
            enhanced_latent: 增强后的潜在表示
            attention_gate: 注意力门控
            class_gate: 类别门控
            fusion_weights: 融合权重
        """
        batch_size, channels, height, width = depth_latent.shape

        # 1. 提取公共纹理特征（复用基础编码器）
        tex_features = self.texture_encoder(texture_features)

        # 2. 注意力引导扩散
        att_enhanced, att_gate, att_weights = self.attention_branch(
            depth_latent, texture_features, attention_map,
            uncertainty_map, num_iterations
        )

        # 3. 类别条件扩散
        cat_enhanced, cat_gate, cat_condition = self.category_branch(
            depth_latent, texture_features, class_labels,
            class_confidence, num_iterations
        )

        # 4. 动态融合权重
        meta_weights = self.meta_fusion(tex_features)  # [B, 2, 1, 1]
        weight_att = meta_weights[:, 0:1, :, :] * self.fusion_weight_attention
        weight_cat = meta_weights[:, 1:2, :, :] * self.fusion_weight_category

        # 5. 融合
        enhanced = weight_att * att_enhanced + weight_cat * cat_enhanced

        return enhanced, att_gate, cat_gate, (weight_att, weight_cat)


class CompleteConditionedDiffusionPipeline(nn.Module):
    """
    完整的条件纹理扩散流程，集成到分割模型中
    """

    def __init__(self, latent_dim=24, window_size=7, num_classes=6, seg_channels = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.seg_channels = seg_channels
        # 深度图编码器
        self.depth_to_latent = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=3, padding=1)
        )

        # 混合条件纹理扩散
        self.conditioned_diffusion = HybridConditionedTextureDiffusion(
            latent_dim=latent_dim,
            window_size=window_size,
            num_classes=num_classes
        )

        # 潜在表示解码器
        self.latent_to_depth = nn.Sequential(
            nn.Conv2d(latent_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

        # 深度图融合模块（用于与分割模型集成）
        self.depth_fusion = nn.Sequential(
            nn.Conv2d(latent_dim + seg_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )

    def forward(self, depth_map, texture_features, attention_map,
                class_labels, segmentation_features=None):
        """
        Args:
            depth_map: 原始深度图 [B, 1, H, W]
            texture_features: 纹理特征 [B, 3, H, W]
            attention_map: 注意力图 [B, 1, H, W]
            class_labels: 类别标签
            segmentation_features: 分割模型的特征 [B, 256, H, W] (可选)

        Returns:
            enhanced_depth: 增强后的深度图
            enhanced_latent: 增强后的潜在表示
            att_gate: 注意力门控
            cat_gate: 类别门控
            fused_features: 融合后的特征（用于分割）
        """
        # 1. 深度图编码
        depth_latent = self.depth_to_latent(depth_map)

        # 2. 条件纹理扩散
        enhanced_latent, att_gate, cat_gate, fusion_weights = self.conditioned_diffusion(
            depth_latent, texture_features, attention_map, class_labels
        )

        # 3. 解码回深度图
        enhanced_depth = self.latent_to_depth(enhanced_latent)

        # 4. 如果提供了分割特征，进行融合
        if segmentation_features is not None:
            # 确保尺寸匹配
            if segmentation_features.shape[-2:] != enhanced_latent.shape[-2:]:
                enhanced_latent = F.interpolate(
                    enhanced_latent,
                    size=segmentation_features.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            fused_features = self.depth_fusion(
                torch.cat([segmentation_features, enhanced_latent], dim=1)
            )
        else:
            fused_features = None

        return enhanced_depth, enhanced_latent, att_gate, cat_gate, fused_features


# 示例用法
if __name__ == "__main__":
    batch_size = 4
    latent_dim = 24
    height, width = 28, 28
    num_classes = 6

    # 创建输入
    depth_map = torch.randn(batch_size, 1, height, width)
    texture = torch.randn(batch_size, 3, height, width)
    attention_map = torch.rand(batch_size, 1, height, width)  # [0,1] 范围
    class_labels = torch.randint(0, num_classes, (batch_size,))
    segmentation_features = torch.randn(batch_size, 256, height, width)

    # print("=" * 50)
    # print("测试注意力引导纹理扩散")
    # print("=" * 50)
    # att_diffusion = AttentionGuidedTextureDiffusion(
    #     latent_dim=latent_dim,
    #     window_size=7
    # )
    # depth_latent = torch.randn(batch_size, latent_dim, height // 2, width // 2)
    # texture_small = F.interpolate(texture, size=(height // 2, width // 2))
    # att_map_small = F.interpolate(attention_map, size=(height // 2, width // 2))
    #
    # enhanced_att, att_gate, att_weights = att_diffusion(
    #     depth_latent, texture_small, att_map_small
    # )
    # print(f"增强后潜在表示形状: {enhanced_att.shape}")
    # print(f"注意力门控形状: {att_gate.shape}")
    # print(f"注意力门控范围: [{att_gate.min():.4f}, {att_gate.max():.4f}]")
    #
    # print("\n" + "=" * 50)
    # print("测试类别条件纹理扩散")
    # print("=" * 50)
    # cat_diffusion = CategoryConditionedTextureDiffusion(
    #     latent_dim=latent_dim,
    #     window_size=7,
    #     num_classes=num_classes
    # )
    #
    # enhanced_cat, class_gate, condition_map = cat_diffusion(
    #     depth_latent, texture_small, class_labels
    # )
    # print(f"增强后潜在表示形状: {enhanced_cat.shape}")
    # print(f"类别门控形状: {class_gate.shape}")
    # print(f"条件融合图形状: {condition_map.shape}")
    #
    # print("\n" + "=" * 50)
    # print("测试混合条件纹理扩散")
    # print("=" * 50)
    # hybrid_diffusion = HybridConditionedTextureDiffusion(
    #     latent_dim=latent_dim,
    #     window_size=7,
    #     num_classes=num_classes
    # )
    #
    # enhanced_hybrid, att_gate2, cat_gate2, fusion_weights = hybrid_diffusion(
    #     depth_latent, texture_small, att_map_small, class_labels
    # )
    # print(f"增强后潜在表示形状: {enhanced_hybrid.shape}")
    # print(f"融合权重 - attention: {fusion_weights[0].mean().item():.4f}, "
    #       f"category: {fusion_weights[1].mean().item():.4f}")
    #
    # print("\n" + "=" * 50)
    print("测试完整条件扩散流程")
    print("=" * 50)
    condTxd = CompleteConditionedDiffusionPipeline(
        latent_dim=latent_dim,
        window_size=7,
        num_classes=num_classes
    )

    enhanced_depth, enhanced_latent, att_gate3, cat_gate3, fused_features = condTxd(
        depth_map, texture, attention_map, class_labels, segmentation_features
    )
    print(f"增强后深度图: {enhanced_depth.shape}")
    print(f"增强后潜在表示: {enhanced_latent.shape}")
    print(f"融合特征: {fused_features.shape}")