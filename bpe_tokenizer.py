import json





# BPE (Byte Pair Encoding) 分词器实现
# 核心思想：通过统计相邻 token 对的出现频率，逐步将高频对合并为新 token，从而学习子词词汇表
class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []
        self.id_to_token = {}

    def get_stats(self, tokens):
        """统计相邻 token 对（bigram）的出现频率
        
        Args:
            tokens: 字符串列表，例如 ["a", "a", "a", "b"]
        Returns:
            stats: 字典，key 为 (左token, 右token) 元组，value 为该对出现的次数
            例如 {("a","a"): 2, ("a","b"): 1}
        """
        stats = {}

        # 遍历所有相邻的 token 对，统计每对出现的次数
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])

            if pair not in stats:
                stats[pair] = 0
            stats[pair] += 1

        return stats

    def merge(self, tokens, pair, new_token):
        """将 tokens 列表中所有出现的 pair 对替换合并为 new_token
        
        Args:
            tokens: 原始 token 列表
            pair: 要被合并的相邻 token 对，例如 ("a", "a")
            new_token: 合并后生成的新 token，例如 "aa"
        Returns:
            result: 合并后的新 token 列表
            例如 ["a","a","a","b"] 合并 ("a","a") 后 → ["aa", "a", "b"]
        """
        result = []
        i = 0
        while i < len(tokens):
            # 如果当前位置和下一个位置恰好匹配要合并的 pair，则替换为 new_token
            if (i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair):
                result.append(new_token)
                i += 2  # 跳过已合并的两个 token
            else:
                result.append(tokens[i])
                i += 1  # 不匹配则保留原 token，前进一位
        return result

    def get_best_pair(self, stats):
        """从频率统计表中找出出现次数最多的 token 对（即最优合并对）
        
        Args:
            stats: get_stats() 返回的频率字典
        Returns:
            best_pair: 频率最高的 (token1, token2) 元组，若为空则返回 None
        """
        best_pair = None
        max_count = 0

        # 遍历所有 pair，找出计数最大的那个
        for pair, count in stats.items():
            if count > max_count:
                best_pair = pair
                max_count = count
        return best_pair

    def train(self, tokens, num_merges):
        # 1. 保存最初的 token，去除重复项但保留原顺序
        vocab_tokens = list(dict.fromkeys(tokens))

        # 2. 保存训练过程中的合并规则
        merges = []

        for _ in range(num_merges):
            # 3. 统计相邻 token 对
            stats = self.get_stats(tokens)

            if not stats:
                break

            # 4. 找到出现次数最多的 token 对
            pair = self.get_best_pair(stats)

            # 5. 生成新的 token
            new_token = "".join(pair)

            # 6. 执行合并
            tokens = self.merge(tokens, pair, new_token)

            # 7. 记录合并规则
            merges.append((pair, new_token))

            # 8. 将新 token 加入候选词表（避免重复）
            if new_token not in vocab_tokens:
                vocab_tokens.append(new_token)

        # 9. 保存合并规则
        self.merges = merges

        # 10. 建立 token -> ID 的映射
        self.vocab = {
            token: idx
            for idx, token in enumerate(vocab_tokens)
        }

        # 11. 建立 ID -> token 的映射
        self.id_to_token = {
            idx: token
            for token, idx in self.vocab.items()
        }

        return tokens

    def train_from_text(self, text, num_merges):
        # 第一步：将文本拆成单个字符
        tokens = list(text)

        # 第二步：调用已经实现的 train() 方法
        return self.train(tokens, num_merges)

    def encode(self, text):
        # 第 1 步：把文本拆成单个字符
        tokens = list(text)

        # 第 2 步：按照训练时学到的顺序执行合并
        for pair, new_token in self.merges:
            tokens = self.merge(tokens, pair, new_token)

        # 第 3 步：把每个 token 转换成数字 ID
        token_ids = []

        for token in tokens:
            if token not in self.vocab:
                raise ValueError(f"未知 token: {token}")

            token_ids.append(self.vocab[token])

        # 第 4 步：返回数字列表
        return token_ids

    def decode(self, ids):
        # 第 1 步：准备一个空列表，用来保存 token
        tokens = []

        # 第 2 步：遍历每一个数字 ID
        for token_id in ids:

            # 第 3 步：检查这个 ID 是否存在于词表中
            if token_id not in self.id_to_token:
                raise ValueError(f"未知 token ID: {token_id}")

            # 第 4 步：根据 ID 找到对应的 token
            token = self.id_to_token[token_id]

            # 第 5 步：把 token 保存起来
            tokens.append(token)

        # 第 6 步：把所有 token 拼接成一个字符串
        return "".join(tokens)

    def save(self, path):
        # 准备需要保存的数据
        data = {
            "vocab": self.vocab,
            "merges": [
                [list(pair), new_token]
                for pair, new_token in self.merges
            ]
        }

        # 将数据写入 JSON 文件
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        # 从 JSON 文件读取数据
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 恢复词表
        self.vocab = data["vocab"]

        # 恢复反向词表
        self.id_to_token = {
            idx: token
            for token, idx in self.vocab.items()
        }

        # 将合并规则中的 pair 恢复成元组
        self.merges = [
            (tuple(pair), new_token)
            for pair, new_token in data["merges"]
        ]

        return self




if __name__ == "__main__":
    # 第一次：训练并保存
    tokenizer = BPETokenizer()
    tokenizer.train_from_text("aaab", 2)

    tokenizer.save("bpe_tokenizer.json")
    print("分词器已保存")

    # 第二次：创建新对象并加载
    tokenizer2 = BPETokenizer()
    tokenizer2.load("bpe_tokenizer.json")

    # 验证加载后的分词器
    ids = tokenizer2.encode("aaab")
    print("编码结果：", ids)
    print("解码结果：", tokenizer2.decode(ids))



class ByteBPETokenizer:
    def __init__(self):
        # 1. 创建 256 个基础字节 token
        self.vocab = {
            bytes([i]): i
            for i in range(256)
        }

        # 2. 保存合并规则
        self.merges = []

        # 3. 创建反向词表：ID -> token
        self.id_to_token = {
            idx: token
            for token, idx in self.vocab.items()
        }

    def get_stats(self, tokens):
        """统计相邻 token 对出现的次数"""
        stats = {}

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            stats[pair] = stats.get(pair, 0) + 1

        return stats

    def get_best_pair(self, stats):
        """找到出现次数最多的相邻 token 对"""
        best_pair = None
        max_count = 0

        for pair, count in stats.items():
            if count > max_count:
                best_pair = pair
                max_count = count

        return best_pair

    def merge(self, tokens, pair, new_token):
        """将指定的相邻 token 对合并"""
        result = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        return result

    def train(self, text, num_merges):
        """使用文本训练字节级 BPE"""
        # 将文本转换为 UTF-8 字节 token
        tokens = [
            bytes([b])
            for b in text.encode("utf-8")
        ]

        # 保存初始的 256 个基础 token
        vocab_tokens = [
            bytes([i])
            for i in range(256)
        ]

        merges = []

        for _ in range(num_merges):
            stats = self.get_stats(tokens)

            if not stats:
                break

            pair = self.get_best_pair(stats)
            new_token = pair[0] + pair[1]

            tokens = self.merge(tokens, pair, new_token)
            merges.append((pair, new_token))

            if new_token not in vocab_tokens:
                vocab_tokens.append(new_token)

        # 重建词表，基础字节 ID 保持为 0～255
        self.vocab = {
            token: idx
            for idx, token in enumerate(vocab_tokens)
        }

        self.id_to_token = {
            idx: token
            for token, idx in self.vocab.items()
        }

        self.merges = merges

        return tokens

    def encode(self, text):
        """将文本转换成 token ID 列表"""
        tokens = [
            bytes([b])
            for b in text.encode("utf-8")
        ]

        for pair, new_token in self.merges:
            tokens = self.merge(tokens, pair, new_token)

        return [self.vocab[token] for token in tokens]

    def decode(self, ids):
        """将 token ID 列表还原为文本"""
        token_bytes = b"".join(
            self.id_to_token[token_id]
            for token_id in ids
        )

        return token_bytes.decode("utf-8")