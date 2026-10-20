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


# 测试入口
# if __name__ == "__main__":
#     tokenizer = BPETokenizer()
#
#     # 测试1：统计相邻 token 对频率
#     tokens = ["a", "a", "a", "b"]
#     result = tokenizer.get_stats(tokens)
#     print(result)  # 输出: {('a', 'a'): 2, ('a', 'b'): 1}
#
#     # 测试2：将最高频的 pair ("a","a") 合并为新 token "aa"
#     tokens = ["a", "a", "a", "b"]
#     result = tokenizer.merge(tokens, ("a", "a"), "aa")
#     print(result)  # 输出: ['aa', 'a', 'b']
#     tokens = ["a", "b", "a", "b"]
#     result = tokenizer.train(tokens, num_merges=1)
#     print(result)
# 