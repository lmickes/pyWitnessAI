import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F


class CLIPFillerSelector:
    def __init__(self, image_dir: str, device: str = None):
        """
        param image_dir: 存放候选照片的目录
        param device:    'cuda' 或 'cpu'，默认自动检测
        """
        self.image_dir = image_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 加载 CLIP 模型和预处理函数
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # 2. 构建图像路径列表
        self.image_paths = [
            os.path.join(image_dir, fn)
            for fn in os.listdir(image_dir)
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        # 3. 预先提取并缓存所有图像特征
        self._cache_image_features()

    def _cache_image_features(self):
        # 对所有照片进行预处理并批量提取 CLIP 图像特征向量。
        all_feats = []
        batch_size = 32
        for i in tqdm(range(0, len(self.image_paths), batch_size), desc="Encoding images"):
            batch_imgs = []
            for p in self.image_paths[i: i + batch_size]:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(self.preprocess(img))
            batch_tensor = torch.stack(batch_imgs, dim=0).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch_tensor)
                feats = F.normalize(feats, dim=-1)  # 单位化
            all_feats.append(feats)
        self.image_features = torch.cat(all_feats, dim=0)  # [N, D]

    def query(self, verbal_desc: str, top_k: int = 5):
        """
        根据口头描述检索最相似的图像
        param verbal_desc: 目击者描述（自然语言）
        param top_k:       返回的照片数量
        return:            List[(image_path, score)]（按相似度降序）
        """
        # 1. 文本编码
        text_tokens = clip.tokenize([verbal_desc], truncate=True).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(text_tokens)
            text_feat = F.normalize(text_feat, dim=-1)  # [1, D]

        # 2. 计算余弦相似度 scores: [N]
        scores = (self.image_features @ text_feat.T).squeeze(1)

        # 3. 取 Top-K
        topk_vals, topk_idxs = scores.topk(top_k, largest=True)
        results = [
            (self.image_paths[idx], float(topk_vals[i].item()))
            for i, idx in enumerate(topk_idxs)
        ]
        return results


if __name__ == "__main__":
    selector = CLIPFillerSelector("D:/PhD/Studies/VerbalDescription/FillerPoolColloff")
    query_str = "30 years old, white male, short hair"
    # query_str = "A middle-aged man with short hair and black framed glasses"
    top_matches = selector.query(query_str, top_k=8)
    for img_path, score in top_matches:
        print(f"{img_path}  —— 相似度：{score:.4f}")
