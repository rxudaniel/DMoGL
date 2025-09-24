import re
import jieba
import jieba.posseg as pseg
import pandas as pd
import time  # 添加时间模块
from 文本相似性_中文RoBERTa import calculate_similarity        # 优化部分-语义相似函数


# 自定义词典文件名为 'user_dict.txt'，并且位于当前目录下
jieba.load_userdict('/share/home/pengwenzhong/hxh/Deeptest/Task1/TIYU_dic.txt')
def main():
    csv_file_path = "TIYU_15000_2025_0417 .csv"  # 替换为你的CSV文件路径

    # 读取CSV文件
    try:
        # 使用pandas读取CSV文件
        df = pd.read_csv(csv_file_path, encoding='GBK', header=None)
        sentences = df[0].tolist()  # 假设句子在第一列
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sentences = []  # 如果发生错误，返回一个空列表

        # 创建邻接表对象
    adjacency_list = AdjacencyList()
    adjacency_list.process_and_build(sentences)
    # 显示构建结果
    print("\n邻接表:")                             ##调试1
    adjacency_list.display()

    # 创建热词提取器
    extractor = HotWordsExtractor(adjacency_list)
    while True:
        # 动态输入 n1 和 n2
        try:
            n1 = int(input("\n请输入初始支持度阈值 n1（-1 退出程序）: "))
            if n1 == -1:
                break

            n2 = int(input("请输入过滤支持度阈值 n2（-1 退出程序）: "))
            if n2 == -1:
                break
        except ValueError:
            print("请输入有效的整数值！")
            continue

        # 全局频繁模式集合
        global_hot_words = {}

        # 记录总运行时间
        total_start_time = time.time()

        # 记录单个通道的开始时间
        start_time = time.time()

        # 提取当前通道的热词集合
        extractor.hot_words = {}       # 重置热词集合
        extractor.extract(n1, 0)
        # 过滤通道的热词
        extractor.filter_hot_words(n2)
        #输出热词
        extractor.display_hot_words()  ####调试输出111
        # 合并到全局集合，以便后续使用
        extractor.merge_hot_words(extractor.hot_words, global_hot_words)
        # 记录单个通道的结束时间
        end_time = time.time()
        print(f"通道花费时间: {end_time - start_time:.2f} 秒")



        phrasemerger1=PhraseMerger(global_hot_words,adjacency_list)
        phrasemerger1.merge_included_phrases()
        # phrasemerger1.display_hot_words()      ####调试输出222
        phrasemerger1.merge_highly_similar_sources()
        phrasemerger1.display_hot_words()

        global_hot_words1=phrasemerger1.global_hot_words
        Treehot=HotPhraseExtractor(global_hot_words1,adjacency_list)
        Treehot.build_tree()
        Treehot.check_and_adjust_bottom_layer()
        Treehot.remove_weak_nodes_by_threshold(n2)
        hot_word=Treehot.extract_hot_phrases()
        Treehot.display_forest()
        Treehot.display_hot_phrases(hot_word)
        # print(sentences)
        sub_processor = SubCorpusProcessor(sentences, Treehot, n1, n2)
        sub_processor.process_sub_corpora()

        global_hot_words.clear()
        # 记录总时间
        total_end_time = time.time()
        print(f"\n总运行时间: {total_end_time -total_start_time:.2f} 秒")
class WordNode:
    """
    表示词典表中的一个词项节点。
    """
    id=0
    def __init__(self, word,pos=None):
        self.word = word                 # 词项
        self.pos = pos                   # 词性，初始化为 None
        self.num = 0                     # 支持度
        self.fir = "None"                # 首条语料指针 (H(i,j))，初始化为 None
        self.end = "None"                # 最后语料指针 (H(i,j))，初始化为 None
        self.ID=WordNode.id              #词项编号
        WordNode.id+=1


class CorpusNode:
    """
    表示语料表中的一个语料节点。
    """
    def __init__(self, word, sentence_index, word_index,pos):
        self.word = word                 # 当前语料的词项
        self.pos = pos                   # 词性
        self.position = f"H({sentence_index},{word_index})"  # 当前语料位置 H(i,j)
        self.next_occurrence = "None"    # 下次出现该词项的语料节点，初始化为 None
        self.prev_word = "None"          # 当前语料中上一个词项，初始化为 None
        self.next_word = "None"          # 当前语料中下一个词项，初始化为 None



class AdjacencyList:
    """
    语料-词典邻接表构建类。
    """
    def __init__(self):
        self.word_dict = {}              # 词典表 WS：键为词项，值为对应的 WordNode
        self.corpus_nodes = []           # 语料表 HS：存储 CorpusNode
        # self.simple_words = set()        # 特定词性的词集合

    def process_and_build(self, sentences1):
        """
        分词、过滤并直接构建语料-词典邻接表。
        :param sentences1: 原始语料列表
        """
        # 定义正则表达式，用于去除标点符号
        def remove_punctuation(text):
            return re.sub(r'[^\w\s]', '', text)  # 保留字母、数字、下划线和空白字符

        # 预处理：去除标点符号
        cleaned_sentences = [remove_punctuation(sentence) for sentence in sentences1]
        for i, sentence in enumerate(cleaned_sentences):             # 遍历每条语料
            prev_node = None                                  # 初始化当前语料中上一个词项节点
            # 使用 jieba 分词并获取词性
            # 添加自定义词典条目

            term_list = pseg.cut(sentence)
            word_index = 0                                   # 当前语料中词项的位置计数
            for term in term_list:
                word = term.word
                pos = term.flag
                # 过滤掉标点符号
                if pos in ["x", "w"]:                        # 'x' 和 'w' 通常表示标点符号
                    continue

                # 创建语料节点 H(i,j)
                current_corpus_node = CorpusNode(word, i, word_index, pos)
                word_index += 1

                # 将当前语料节点添加到语料表
                self.corpus_nodes.append(current_corpus_node)

                # 设置语料节点的语料中上下文关系
                if prev_node:
                    prev_node.next_word = current_corpus_node.position
                    current_corpus_node.prev_word = prev_node.position
                prev_node = current_corpus_node

                # 更新词典表 WS
                if word not in self.word_dict:
                    # 如果词项不在词典表中，创建词典节点
                    self.word_dict[word] = WordNode(word, pos)

                word_node = self.word_dict[word]
                word_node.num += 1  # 更新支持度

                if word_node.fir == "None":
                    # 设置首条语料指针
                    word_node.fir = current_corpus_node.position
                else:
                    # 更新最后一次出现的语料的 next_occurrence
                    last_corpus_node = next(
                        node for node in self.corpus_nodes if node.position == word_node.end
                    )
                    last_corpus_node.next_occurrence = current_corpus_node.position

                # 更新最后语料指针
                word_node.end = current_corpus_node.position

    def display(self):
        """
        打印词典表 WS 和语料表 HS。
        """
        print("词典表（Word Dictionary，WS）:")
        for word, node in self.word_dict.items():
            if node.num>2:
                print(f"词项: {word}, 支持度: {node.num}, 首条语料: {node.fir}, 最后语料: {node.end}")

        print("\n语料表（Corpus List，HS）:")
        for corpus_node in self.corpus_nodes:
            print(f"{corpus_node.position}: 词项: {corpus_node.word}, 下次出现: {corpus_node.next_occurrence}, "
                  f"语料中上一个词项: {corpus_node.prev_word}, 语料中下一个词项: {corpus_node.next_word}")

class HotWordsExtractor:
    """
    频繁模式提取器，根据邻接表提取热词。
    """
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list
        self.hot_words = {}                    # 当前通道的频繁模式集合：key 是词组，value 是 (支持度, 共情来源句子集合)
        self.views = {}                        # 滑动窗口集合：key 是语料编号，value 是 {'s', 'e', 'state'}

    def extract(self, n1, mode=0):
        """
        按照算法提取频繁模式，支持度小于 n1 的词将被忽略。
        :param n1: 支持度阈值
        :param mode: 遍历模式，0-从前往后，1-从后往前，2-从中间往后再往前
        """
        words = list(self.adjacency_list.word_dict.items())

        # 根据 mode 确定遍历顺序
        if mode == 1:
            words = list(reversed(words))
        elif mode == 2:
            mid = len(words) // 2
            words = words[mid:] + words[:mid][::-1]

        # 初始化滑动窗口
        self.views = {i: {'s': None, 'e': None, 'state': 'silent'} for i in
                      range(len(self.adjacency_list.corpus_nodes))}

        # 初始化上一轮的活跃窗口集合 NS
        prev_active_views = set()

        # 遍历词典表中的词
        for word, word_node in words:
            # 跳过支持度低于阈值的词
            if word_node.num < n1 :
                continue

            # 当前轮次的激活/延续窗口集合 NE
            current_active_views = set()

            # 遍历当前词项的所有共情点
            current_pos = word_node.fir
            while current_pos != "None":
                # 获取当前语料节点
                corpus_node = next(
                    (node for node in self.adjacency_list.corpus_nodes if node.position == current_pos),
                    None                                   # 默认值，表示找不到时返回 None
                )
                sentence_idx = int(corpus_node.position.split("(")[1].split(",")[0])

                # 当前语料的滑动窗口
                view = self.views[sentence_idx]

                # 滑动窗口状态处理
                if view['state'] == 'silent':
                    # 新入型：激活窗口
                    view['s'] = corpus_node                # 设置起始位置
                    view['e'] = corpus_node                # 设置结束位置
                    view['state'] = 'active'
                    current_active_views.add(sentence_idx)
                    # 检查是否与当前词项共情
                elif view['state'] == 'active' and corpus_node.word == word:
                        # 延续型：更新窗口结束点
                        view['e'] = corpus_node
                        current_active_views.add(sentence_idx)

                # 获取下一个共情点
                current_pos = corpus_node.next_occurrence

            # 处理上一轮存在但未在当前轮次延续的窗口（即发生截断的窗口）
            to_reset_views = []  # 用于记录本轮需要重置的窗口
            for prev_idx in prev_active_views:
                if prev_idx not in current_active_views:
                    # print("缺少", prev_idx)                ##调试2
                    # print(self.views[prev_idx])           ##调试3
                    # 调用 _process_cutoff 处理截断窗口
                    # 将窗口记录下来，稍后统一关闭
                    to_reset_views.append(prev_idx)
                    # 调用 extract_View 处理截断窗口
            if len(to_reset_views)>=1:
                self.extract_View(to_reset_views)

            # 统一关闭被截断的窗口
            for reset_idx in to_reset_views:
                self.views[reset_idx]['s'], self.views[reset_idx]['e'], self.views[reset_idx][
                    'state'] = None, None, 'silent'

            # 更新上一轮的活跃窗口集合
            prev_active_views = current_active_views

        # 处理所有活跃窗口（结束时）
        for sentence_idx, view in self.views.items():
            if view['state'] == 'active':
                # self._process_cutoff(sentence_idx,  view)
                # 重置窗口
                view['s'], view['e'], view['state'] = None, None, 'silent'

        return self.hot_words

    def extract_View(self, cutoff_indices):
        """
        处理截断型窗口，挑选出文本跨度最长的窗口。
        :param cutoff_indices: 被截断的窗口编号列表
        """
        # 找到具有最前沿 start 词项位置的截断窗口
        min_start_view = None
        min_start_idx = None
        for cutoff_idx in cutoff_indices:
            view = self.views[cutoff_idx]
            if not view['s'] or not view['e']:
                continue
            if min_start_view is None or view['s'].position < min_start_view['s'].position:
                min_start_view = view
                min_start_idx = cutoff_idx
        # print(min_start_idx)                 ##调试10
        # print(min_start_view)                ##调试11
        # 如果找到了有效的窗口，则进行处理
        if min_start_view:
            self._process_cutoff(min_start_idx, min_start_view)

    def _process_cutoff(self, cutoff_idx,  cutoff_view):
        """
        处理截断型窗口，基于其他活跃窗口提取频繁模式。
        :param cutoff_idx: 被截断的窗口编号
        :param current_active_views: 当前活跃窗口集合
        :param cutoff_view: 被截断的窗口
        """
        cutoff_start = cutoff_view['s']
        cutoff_end = cutoff_view['e']

        if not cutoff_start or not cutoff_end:
            return

        # 检查其他活跃窗口
        for other_idx, other_view in self.views.items():
            if other_idx == cutoff_idx or other_view['state'] != 'active':
                continue                                # 跳过自己或非活跃窗口

            # print("第",other_idx,"个窗口来做比较了")      ##调试信息4
            other_view = self.views[other_idx]
            other_start = other_view['s']
            other_end = other_view['e']
            # print(other_start.word)                   ##调试5
            # print(other_end.word)                     ##调试6

            if not other_start or not other_end:
                continue
            # 根据词项在词典表中的顺序提取频繁模式
            start_word_idx = self._get_word_index(cutoff_start.word)
            cutoff_end_word_index=self._get_word_index(cutoff_end.word)
            other_start_word_idx = self._get_word_index(other_start.word)
            other_end_word_index=self._get_word_index(other_end.word)
            # print(start_word_idx)
            # print(cutoff_end_word_index)              #调试信息7
            # print(other_start_word_idx)
            # print(other_end_word_index)

            if start_word_idx >= other_start_word_idx:
                # W_A(Viewi.s) <= W_A(Viewq.s)
                # phrase = self._extract_phrase(other_start_word_idx, cutoff_end_word_index)
                i=other_start_word_idx
                while(i<=cutoff_end_word_index):
                    phrase = self._extract_phrase(i, cutoff_end_word_index)
                    # print(phrase)                      ##调试信息8
                    # 更新频繁模式集合
                    self._update_hot_words(phrase)
                    i=i+1
            else:
                # W_A(Viewi.s) > W_A(Viewq.s)
                # phrase = self._extract_phrase(other_start_word_idx, other_end_word_index)
                j = start_word_idx
                while (j <= cutoff_end_word_index):
                    phrase = self._extract_phrase(j, cutoff_end_word_index)
                    # print(phrase)                     ##调试信息9
                    # 更新频繁模式集合
                    self._update_hot_words(phrase)
                    j=j+1

            # # 更新频繁模式集合
            # self._update_hot_words(phrase)


    def _get_word_index(self, word):
        """
        获取词项在词典表中的索引。
        :param word: 词项
        :return: 索引（int），如果词项不存在则返回 -1
        """
        word_list = list(self.adjacency_list.word_dict.keys())
        return word_list.index(word) if word in word_list else -1

    def _extract_phrase(self, start, end):
        """
        从词典表中提取从 start 到 end 对应的连续词项（包括 start 和 end）。
        :return: 提取的词项序列（字符串，按词典表中的顺序合并，没有空格）
        """
        # 获取词典表中所有词项的顺序
        word_list = list(self.adjacency_list.word_dict.keys())


        # 提取从 start 到 end 的连续词项，并合并为字符串
        phrase = "".join(word_list[start:end + 1])

        # 定义特定词性的集合
        single_word_exclude_pos = ['adv','ry', 'rr', 'r', 'uj', 'ul', 'u', 'd', 'f', 'p', 'c', 'a','q','zg','m','t','y']
        multi_word_start_pos_exclude = ['vshi', 'uj', 'ul', 'vyou']

        # 判断是否只有一个词项
        if start == end:
            # 获取单个词项的词性
            word_node = self.adjacency_list.word_dict[word_list[start]]
            if word_node.pos in single_word_exclude_pos:
                return None
        else:
            # 获取起始词项的词性
            start_word_node = self.adjacency_list.word_dict[word_list[start]]
            if start_word_node.pos in multi_word_start_pos_exclude:
                return None

        # 如果通过了上述所有判断，则提取该频繁模式
        return phrase


    def _update_hot_words(self, phrase):
        """
        更新热词集合，计算支持度和来源。
        :param phrase: 提取的频繁模式
        """
        if not phrase or phrase in "，。！？；：（）【】“”‘’《》、":
            return

        # 如果 phrase 已经存在于热词集合，直接跳过
        if phrase in self.hot_words:
            return
        phrase1=str(phrase)
        # 分词获取 phrase 的第一个词项及其长度
        phrase_words = list(pseg.cut(phrase1))

        # 获取第一个词项
        if phrase_words:
            phrase_first_word = phrase_words[0].word
            # print("第一个词项:", phrase_first_word)          #调试12
        else:
            print("没有分词结果")

        # 获取词项数量
        phrase_length = len(phrase_words)
        # print("词项数量:", phrase_length)                   ##调试13

        # 如果第一个词项不在词典表中，直接返回
        if phrase_first_word not in self.adjacency_list.word_dict:
            return

        # 获取第一个词项的词典节点
        word_node = self.adjacency_list.word_dict[phrase_first_word]

        # 遍历该词项的所有共情点
        current_pos = word_node.fir
        support_count = 0
        sources = set()

        while current_pos != "None":
            # 获取当前共情点对应的语料节点
            corpus_node = next(
                (node for node in self.adjacency_list.corpus_nodes if node.position == current_pos),
                None
            )

            if not corpus_node:
                break

            # 获取语料编号和当前词项在语料中的位置
            sentence_idx = int(corpus_node.position.split("(")[1].split(",")[0])
            word_idx = int(corpus_node.position.split(",")[1].split(")")[0])
            # print(word_idx)                               #调试14

            # 提取当前语料的所有词
            sentence_words = [
                node.word for node in self.adjacency_list.corpus_nodes
                if int(node.position.split("(")[1].split(",")[0]) == sentence_idx
            ]
            # print(sentence_words)                        #调试15

            # 检查从当前位置开始的长度是否足够
            if word_idx + phrase_length <= len(sentence_words):
                # 提取与 phrase 长度相同的片段
                candidate_phrase = "".join(sentence_words[word_idx:word_idx + phrase_length])
                # print(candidate_phrase)                   #调试16
                # 如果匹配成功，更新支持度和来源
                if candidate_phrase == phrase:
                    support_count += 1
                    sources.add(sentence_idx)

            # 获取下一个共情点
            current_pos = corpus_node.next_occurrence

        # 将结果存入热词集合
        self.hot_words[phrase] = [support_count, sources]

    def filter_hot_words(self, n2):
        """
        根据支持度阈值 n2 过滤频繁模式集合。
        :param n2: 支持度过滤阈值
        """
        self.hot_words = {
            phrase: (count, sources)
            for phrase, (count, sources) in self.hot_words.items()
            # if count >= n2 and len(phrase)>2
            if count >= n2
        }

    def merge_hot_words(self, channel_hot_words, global_hot_words):
        """
        将某个通道提取的频繁模式集合合并到全局集合中。
        如果频繁模式已存在，则忽略。
        :param channel_hot_words: 某个通道的频繁模式集合
        :param global_hot_words: 全局频繁模式集合
        """
        for phrase, (count, sources) in channel_hot_words.items():
            if phrase in global_hot_words:
                # 如果已经存在，跳过，不合并
                continue
            else:
                # 如果不存在，直接添加
                global_hot_words[phrase] = [count, sources]


    def display_hot_words(self):
        """
        打印频繁模式集合。
        """
        print("\n频繁模式（W）：")
        for phrase, (count, sources) in self.hot_words.items():
            print(f"频繁模式: {phrase}, 支持度: {count}, 共情来源: {sorted(sources)}")

class PhraseMerger:
    def __init__(self, global_hot_words,adjacency_list):
        self.global_hot_words = global_hot_words        # 存储频繁模式phrase的集合
        self.adjacency_list = adjacency_list  # 持有 AdjacencyList 的实例

    def calculate_similarity(self, sentence1, sentence2):
        # 导入calculate_similarity函数，计算语义相似度
        return calculate_similarity(sentence1, sentence2)

    def merge_similar_phrases(self):
        # 获取所有phrase的列表
        phrases = list(self.global_hot_words.keys())

        # 存储需要删除的phrase
        to_delete = set()

        # 存储合并后的结果
        merged_results = {}

        # 遍历所有phrase
        for i, phrase1 in enumerate(phrases):
            if phrase1 in to_delete:
                continue  # 如果phrase1已经被标记为删除，跳过

            for j in range(i + 1, len(phrases)):
                phrase2 = phrases[j]
                if phrase2 in to_delete:
                    continue  # 如果phrase2已经被标记为删除，跳过

                # 计算语义相似度
                similarity = self.calculate_similarity(phrase1, phrase2)
                print(phrase1, "和", phrase2, "语义相似性为：", similarity)

                # 如果语义相似度大于阈值，合并phrase
                if similarity > 0.93:
                    # 调用合并频繁模式的函数
                    merged_phrase = self.merge_phrases(phrase1, phrase2)

                    # 存储合并后的结果
                    merged_results[merged_phrase] = self.increase_support(phrase1, phrase2)

                    # 标记需要删除的phrase
                    to_delete.add(phrase1)
                    to_delete.add(phrase2)
                    break  # 合并后跳出内层循环

        # 遍历结束后，先删除需要删除的phrase
        for phrase in to_delete:
            if phrase in self.global_hot_words:
                del self.global_hot_words[phrase]

        # 将合并后的phrase添加到原频繁模式集合中
        for phrase, value in merged_results.items():
            self.global_hot_words[phrase] = value


    def increase_support(self, phrase1, phrase2):
        """
        增加新来源的支持度，重复来源不增加支持度。
        :param phrase1: 第一个phrase
        :param phrase2: 第二个phrase
        """
        # global_hot_words的结构是：{phrase: (count, sources)}

        count1, sources1 = self.global_hot_words[phrase1]
        count2, sources2 = self.global_hot_words[phrase2]

        # 计算两个phrase来源的并集，并更新支持度
        new_sources = set(sources1) | set(sources2)
        new_count = len(new_sources)

        # 更新global_hot_words中的信息，使用合并后的phrase和新的支持度、来源
        return  (new_count, list(new_sources))


    def merge_included_phrases(self):
        """
        处理两个频繁模式来源相同且存在包含关系时的合并，
        保留最长的，删除较短的。
        """
        # 按phrase长度从长到短排序
        phrases = sorted(self.global_hot_words.keys(), key=lambda x: len(x), reverse=True)
        retained_phrases = set()  # 记录保留的phrase

        for phrase in phrases:
            # 检查是否有更长的phrase包含它且来源相同
            to_delete = False
            for longer_phrase in retained_phrases:
                if phrase in longer_phrase:
                    # 检查来源是否相同
                    _, sources_phrase = self.global_hot_words[phrase]
                    _, sources_longer = self.global_hot_words[longer_phrase]
                    if set(sources_phrase) == set(sources_longer):
                        to_delete = True
                        break
            if not to_delete:
                retained_phrases.add(phrase)

        # 删除未被保留的phrase
        for phrase in list(self.global_hot_words.keys()):
            if phrase not in retained_phrases:
                del self.global_hot_words[phrase]
                # print(f"删除短的phrase: {phrase}")          #调试16

    def calculate_source_similarity(self, sources1, sources2):
        """
        计算两个phrase的来源相似性。
        :param sources1: 第一个phrase的来源列表
        :param sources2: 第二个phrase的来源列表
        :return: 来源相似度
        """
        set1 = set(sources1)
        set2 = set(sources2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        if not union:
            return 0
        return len(intersection) / len(union)

    def merge_highly_similar_sources(self):
        """
        处理来源相似度高的频繁模式，确保多个phrase的来源相似度都高时全部合并。
        """
        # 创建一个临时字典存储新的频繁模式
        new_hot_words = dict(self.global_hot_words)
        to_delete = set()  # 记录需要删除的phrase
        merged_groups = []  # 记录已经合并的组

        # 遍历所有phrase
        for phrase1, (count1, sources1) in self.global_hot_words.items():
            if phrase1 in to_delete:
                continue  # 如果phrase1已经被标记为删除，跳过
            # 创建一个新的合并组
            merged_group = [phrase1]
            merged_sources = set(sources1)
            # 遍历剩余phrase，寻找相似来源的phrase
            for phrase2, (count2, sources2) in self.global_hot_words.items():
                if phrase2 == phrase1 or phrase2 in to_delete:
                    continue
                similarity = self.calculate_source_similarity(sources1, sources2)
                # print(phrase1, "和", phrase2, "来源相似性为：", similarity)   #调试17
                if similarity > 0.95:
                    # 将phrase2加入合并组
                    merged_group.append(phrase2)
                    merged_sources.update(sources2)
                    to_delete.add(phrase2)
            # 如果合并组中有多个phrase，则合并它们
            if len(merged_group) > 1:
                # 逐步合并短语
                merged_phrase = merged_group[0]  # 初始化合并后的短语
                for i in range(1, len(merged_group)):
                    merged_phrase = self.merge_phrases(merged_phrase, merged_group[i])
                # 更新合并后的短语和来源
                new_count = len(merged_sources)
                new_hot_words[merged_phrase] = (new_count, list(merged_sources))
                # 标记合并组中的phrase为删除
                to_delete.update(merged_group)
                # 记录合并组
                merged_groups.append(merged_group)

        # 统一删除标记为删除的phrase
        for phrase in to_delete:
            if phrase in new_hot_words:
                del new_hot_words[phrase]

        # 更新全局频繁模式
        self.global_hot_words = new_hot_words

    def merge_phrases(self, a, b):
        """
        合并两个短语，并按照词典表中的词项顺序拼接，去重。
        :param a: 第一个短语（无分隔符）
        :param b: 第二个短语（无分隔符）
        :return: 合并后的短语
        """

        def split_phrase(phrase):
            """
            根据词典表中的词项切分短语。
            :param phrase: 无分隔符的短语
            :return: 切分后的词项列表
            """
            words = []
            word_dict = self.adjacency_list.word_dict  # 访问 AdjacencyList 的词典表
            max_len = max(len(word) for word in word_dict.keys())  # 词典中最长词项的长度
            i = 0
            while i < len(phrase):
                # 从最大长度开始尝试匹配
                for j in range(min(max_len, len(phrase) - i), 0, -1):
                    candidate = phrase[i:i + j]
                    if candidate in word_dict:
                        words.append(candidate)
                        i += j
                        break
                else:
                    # 如果没有匹配到词项，按单字切分
                    words.append(phrase[i])
                    i += 1
            return words

        # 切分短语 a 和 b
        a_words = split_phrase(a)
        b_words = split_phrase(b)

        # 合并两个词项列表
        combined_words = a_words + b_words

        # 去重并按照词典表中的词项顺序排序
        unique_words = list(dict.fromkeys(combined_words))  # 去重
        # 过滤掉不在词典中的词项
        unique_words = [word for word in unique_words if word in self.adjacency_list.word_dict]
        unique_words.sort(key=lambda word: self.adjacency_list.word_dict[word].ID)  # 按词典表中的ID排序

        # 拼接为最终的短语
        return ''.join(unique_words)

    def display_hot_words(self):
        """
        打印频繁模式集合。
        """
        print("\n频繁模式（W）：")
        # for phrase, (count, sources) in self.global_hot_words.items():
        #     print(f"频繁模式: {phrase}, 支持度: {count}, 共情来源: {sorted(sources)}")
        for phrase, (count, sources) in sorted(self.global_hot_words.items(), key=lambda x: -x[1][0]):
            # if count>1:
              print(f"频繁模式: {phrase}, 支持度: {count}, 共情来源: {sorted(sources)}")

class TreeNode:
    def __init__(self, phrase, support, sources, node_id):
        self.phrase = phrase
        self.support = support
        self.sources = sources
        self.node_id = node_id  # 添加ID属性
        self.children = []
        self.parent = []  # 记录父节点，可能有多个

    def add_child(self, child_node):
        """添加子节点，并确保子节点不重复"""
        if child_node not in self.children:
            self.children.append(child_node)
            # 更新子节点的父节点
            child_node.parent.append(self)

class HotPhraseExtractor:
    def __init__(self, global_hot_words, adjacency_list):
        """
        初始化 HotPhraseExtractor。
        :param global_hot_words: 全局热词列表
        :param adjacency_list: AdjacencyList 的实例
        """
        self.global_hot_words = global_hot_words
        self.adjacency_list = adjacency_list      # 持有 AdjacencyList 的实例
        self.roots = []
        self.node_counter = 0                     # 用于生成唯一ID

    def build_tree(self):
        # 按支持度从高到低排序
        sorted_phrases = sorted(self.global_hot_words.items(), key=lambda x: -x[1][0])

        # 创建所有节点
        nodes = {}
        for phrase, (support, sources) in sorted_phrases:
            node = TreeNode(phrase, support, sources, self.node_counter)
            self.node_counter += 1
            nodes[phrase] = node

        # 构建父子关系
        for parent_phrase, (parent_support, parent_sources) in sorted_phrases:
            parent_node = nodes[parent_phrase]
            for child_phrase, (child_support, child_sources) in sorted_phrases:
                child_node = nodes[child_phrase]
                # 检查是否满足父子关系条件
                if (child_support < parent_support and
                        set(child_sources).issubset(set(parent_sources)) and
                        child_node not in parent_node.children):
                    # 检查是否已经有其他父节点包含当前父节点的来源
                    has_better_parent = False
                    for other_parent_phrase, (other_parent_support, other_parent_sources) in sorted_phrases:
                        if (other_parent_support > parent_support and
                                set(parent_sources).issubset(set(other_parent_sources)) and
                                child_phrase != other_parent_phrase):
                            has_better_parent = True
                            break
                    # 如果没有更好的父节点，则将当前节点添加为子节点
                    if not has_better_parent:
                        # 检查当前父节点的子节点中是否有来源包含的情况
                        to_remove = []
                        for sibling in parent_node.children:
                            if set(child_sources).issubset(set(sibling.sources)):
                                # 如果子节点的来源被某个兄弟节点包含，则将其添加到该兄弟节点下
                                sibling.add_child(child_node)
                                break
                            elif set(sibling.sources).issubset(set(child_sources)):
                                # 如果某个兄弟节点的来源被子节点包含，则将兄弟节点移动到子节点下
                                to_remove.append(sibling)
                                child_node.add_child(sibling)
                        else:
                            # 如果没有来源包含关系，则直接添加为子节点
                            parent_node.add_child(child_node)
                        # 移除已经被移动到子节点下的兄弟节点
                        for sibling in to_remove:
                            parent_node.children.remove(sibling)

        # 找到所有根节点，即没有父节点的节点
        self.roots = []
        for phrase, (support, sources) in sorted_phrases:
            node = nodes[phrase]
            # 检查是否有其他节点包含它的来源，并且支持度比它高
            has_parent = False
            for other_phrase, (other_support, other_sources) in sorted_phrases:
                if (other_support > support and
                        set(sources).issubset(set(other_sources)) and
                        other_phrase != phrase):
                    has_parent = True
                    break
            if not has_parent:
                self.roots.append(node)

    def check_and_adjust_bottom_layer(self):
        """检查并调整每棵树的最底层兄弟节点中的来源包含情况"""
        flag = 1
        while flag:
            flag = 0
            for root in self.roots:
                self._adjust_bottom_layer(root, flag)

    def _adjust_bottom_layer(self, node, flag):
        """递归检查并调整最底层兄弟节点中的来源包含情况"""
        if not node.children:
            return

        # 检查当前节点的子节点是否是最底层
        is_bottom_layer = all(not child.children for child in node.children)

        if is_bottom_layer:
            # 检查兄弟节点中是否存在来源包含情况
            to_remove = set()  # 使用集合来存储需要移除的节点
            for i, child in enumerate(node.children):
                for j, sibling in enumerate(node.children):
                    if i != j and set(child.sources).issubset(set(sibling.sources)):
                        # 如果子节点的来源被某个兄弟节点包含，则将其添加到该兄弟节点下
                        sibling.add_child(child)
                        to_remove.add(child)       # 添加到需要移除的集合中
                        flag = 1
                        break
                    elif i != j and set(sibling.sources).issubset(set(child.sources)):
                        # 如果某个兄弟节点的来源被子节点包含，则将兄弟节点移动到子节点下
                        child.add_child(sibling)
                        to_remove.add(sibling)     # 添加到需要移除的集合中
                        flag = 1
                        break

            # 移除已经被移动到子节点下的兄弟节点
            for child in to_remove:
                if child in node.children:         # 检查是否在列表中
                    node.children.remove(child)

        # 递归检查子节点
        for child in node.children:
            self._adjust_bottom_layer(child, flag)

    def _is_single_word(self, phrase):
        """
        检查短语是否不可再分（即仅为一个词语）。
        """
        words = list(jieba.cut(phrase))
        return len(words) == 1

    def remove_weak_nodes_by_threshold(self, n2):
        """
        从每棵树的最底层叶子节点开始往上遍历，删除满足以下条件之一的节点：
        1. 短语不可再分（即仅为一个词语）且支持度 < n2 + 1。
        2. 新词项数 <= 父节点词项数的 1/3。
        如果某个节点满足删除条件，则删除整棵树中所有 phrase 相同的节点。
        :param n2: 键盘输入的支持度阈值
        """
        # 记录需要删除的 phrase
        phrases_to_remove = set()

        # 第一次遍历：标记需要删除的 phrase
        for root in self.roots:
            self._mark_phrases_to_remove(root, None, n2, phrases_to_remove)

        # 第二次遍历：删除所有标记的 phrase
        for root in self.roots[:]:  # 使用切片复制以避免修改列表时的问题
            self._remove_marked_phrases(root, phrases_to_remove)

    def _mark_phrases_to_remove(self, node, parent, n2, phrases_to_remove):
        """
        递归检查并标记需要删除的 phrase。
        :param node: 当前节点
        :param parent: 当前节点的父节点
        :param n2: 键盘输入的支持度阈值
        :param phrases_to_remove: 需要删除的 phrase 集合
        """
        # 递归处理子节点
        for child in node.children[:]:  # 使用切片复制以避免修改列表时的问题
            self._mark_phrases_to_remove(child, node, n2, phrases_to_remove)

            # 检查当前节点是否需要删除
        if parent is None:  # 根节点单独判断
            if self._is_single_word(node.phrase):
            # 根节点的删除条件：支持度 < n2 + 1
                if node.support < n2 + 1:
                    phrases_to_remove.add(node.phrase)
        else:  # 非根节点
            # 条件一：短语不可再分且支持度 < n2 + 1
            condition1 = self._is_single_word(node.phrase) and node.support < n2 + 1
            # 条件二：新词项数 <= 父节点词项数的 1/3
            condition2 = self._calculate_new_word_count(parent.phrase, node.phrase) <= len(
                list(jieba.cut(parent.phrase))) / 3

            # 如果满足任一条件，则标记需要删除的 phrase
            if condition1 or condition2:
                phrases_to_remove.add(node.phrase)

    def _remove_marked_phrases(self, node, phrases_to_remove):
        """
        递归删除所有标记的 phrase。
        :param node: 当前节点
        :param phrases_to_remove: 需要删除的 phrase 集合
        """
        # 递归处理子节点
        for child in node.children[:]:  # 使用切片复制以避免修改列表时的问题
            self._remove_marked_phrases(child, phrases_to_remove)

        # 删除当前节点的子节点中所有标记的 phrase
        for child in node.children[:]:  # 使用切片复制以避免修改列表时的问题
            if child.phrase in phrases_to_remove:
                self._remove_and_stitch(child, node)

    def _remove_and_stitch(self, node, parent):
        """
        删除当前节点，并将其子节点缝接到父节点。
        :param node: 要删除的节点
        :param parent: 要删除节点的父节点
        """
        # 将当前节点的子节点添加到父节点的子节点列表中
        for child in node.children:
            parent.add_child(child)
        # 从父节点的子节点列表中移除当前节点
        parent.children.remove(node)

    def _calculate_new_word_count(self, parent_phrase, child_phrase):
        """
        计算子节点相较于父节点的新词项数。
        :param parent_phrase: 父节点的短语
        :param child_phrase: 子节点的短语
        :return: 新词项数
        """
        # 分词
        parent_words = set(jieba.cut(parent_phrase))
        child_words = set(jieba.cut(child_phrase))

        # 计算新词项
        new_words = child_words - parent_words
        return len(new_words)


    def merge_phrases(self, a, b):
        """
        合并两个短语，并按照词典表中的词项顺序拼接，去重。
        :param a: 第一个短语（无分隔符）
        :param b: 第二个短语（无分隔符）
        :return: 合并后的短语
        """
        def split_phrase(phrase):
            """
            根据词典表中的词项切分短语。
            :param phrase: 无分隔符的短语
            :return: 切分后的词项列表
            """
            words = []
            word_dict = self.adjacency_list.word_dict              # 访问 AdjacencyList 的词典表
            max_len = max(len(word) for word in word_dict.keys())  # 词典中最长词项的长度
            i = 0
            while i < len(phrase):
                # 从最大长度开始尝试匹配
                for j in range(min(max_len, len(phrase) - i), 0, -1):
                    candidate = phrase[i:i + j]
                    if candidate in word_dict:
                        words.append(candidate)
                        i += j
                        break
                else:
                    # 如果没有匹配到词项，按单字切分
                    words.append(phrase[i])
                    i += 1
            return words

        # 切分短语 a 和 b
        a_words = split_phrase(a)
        b_words = split_phrase(b)

        # 合并两个词项列表
        combined_words = a_words + b_words

        # 去重并按照词典表中的词项顺序排序
        unique_words = list(dict.fromkeys(combined_words))           # 去重
        # 过滤掉不在词典中的词项
        unique_words = [word for word in unique_words if word in self.adjacency_list.word_dict]
        unique_words.sort(key=lambda word: self.adjacency_list.word_dict[word].ID)  # 按词典表中的ID排序

        # 拼接为最终的短语
        return ''.join(unique_words)

    def extract_hot_phrases(self):
        """提取热点词组，满足基本要求和特殊要求"""
        # self.build_tree()
        # self.check_and_adjust_bottom_layer()
        # self.remove_weak_leaf_nodes()
        hot_phrases = {}  # 存储最终的热点词组

        # 单棵树提取操作
        def extract_tree(node, inherited_phrase=""):
            if not node.children:
                # 如果没有子节点，直接提取当前节点短语
                if inherited_phrase:
                    # 如果有继承的短语，拼接去重
                    merged_phrase = self.merge_phrases(inherited_phrase, node.phrase)
                else:
                    # 如果没有继承的短语，直接使用当前节点短语
                    merged_phrase = node.phrase
                # 取当前节点的支持度和来源
                hot_phrases[merged_phrase] = (node.support, node.sources)
            else:
                # 如果有子节点，将当前节点短语与继承的短语拼接去重
                if inherited_phrase:
                    merged_phrase = self.merge_phrases(inherited_phrase, node.phrase)
                else:
                    merged_phrase = node.phrase
                if merged_phrase:
                    # 取当前节点的支持度和来源
                    hot_phrases[merged_phrase] = (node.support, node.sources)
                # 递归处理子节点
                for child in node.children:
                    extract_tree(child, merged_phrase)

        # 遍历森林中的每一棵树，进行提取操作
        for root in self.roots:
            extract_tree(root)
            # 遍历森林中的每一棵树，进行提取操作
            # 后续处理：合并来源相同的短语
        source_to_phrases = {}  # 记录每个来源集合对应的短语列表
        for phrase, (support, sources) in hot_phrases.items():
            # 将来源集合转换为不可变的元组，方便作为字典的键
            source_key = tuple(sorted(sources))
            if source_key not in source_to_phrases:
                source_to_phrases[source_key] = []
            source_to_phrases[source_key].append(phrase)

        # 合并来源相同的短语
        final_hot_phrases = {}
        for source_key, phrases in source_to_phrases.items():
            if len(phrases) > 1:
                # 如果来源相同的短语有多个，递归调用 merge_phrases 合并
                merged_phrase = phrases[0]                        # 初始化为第一个短语
                for phrase in phrases[1:]:
                    merged_phrase = self.merge_phrases(merged_phrase, phrase)
                # 取最后一个短语的支持度和来源
                final_hot_phrases[merged_phrase] = hot_phrases[phrases[-1]]
            else:
                # 如果只有一个短语，直接保留
                final_hot_phrases[phrases[0]] = hot_phrases[phrases[0]]

        return final_hot_phrases

    # def display_tree(self, node, level=0):
    #     """打印树结构"""
    #     print("  " * level + f"Level {level}: Phrase: {node.phrase}, Support: {node.support}, Sources: {node.sources}")
    #     for child in node.children:
    #         self.display_tree(child, level + 1)
    #
    # def display_forest(self):
    #     """打印森林结构"""
    #     print("\n森林结构：")
    #     for root in self.roots:
    #         self.display_tree(root)
    #         print("---")

    def display_tree(self, node, level=0, file=None):
        """将树结构写入文件"""
        if file:
            file.write(
                "  " * level + f"Level {level}: Phrase: {node.phrase}, Support: {node.support}, Sources: {node.sources}\n")
        for child in node.children:
            self.display_tree(child, level + 1, file)

    def display_forest(self):
        """将森林结构写入文件"""
        # with open("实验结果(一万语料)2025_02_20_forest.txt", "w", encoding="utf-8") as file:
        with open("实验结果_剪枝前(体育15000语料_30)_forest.txt", "w", encoding="utf-8") as file:
            file.write("森林结构：\n")
            for root in self.roots:
                self.display_tree(root, file=file)
                file.write("---\n")

    # def display_hot_phrases(self, hot_phrases):
    #     print("\n提取的热点词组：")
    #     for phrase in sorted(hot_phrases.keys()):
    #         support, sources = hot_phrases[phrase]
    #         # if support > 1:
    #         print(f"频繁模式: {phrase}, 支持度: {support}, 共情来源: {sorted(sources)}")

    def display_hot_phrases(self, hot_phrases):
        print("\n提取的热点词组：")

        # 打开文件准备写入
        # with open("实验结果(一万语料)2025_2_20_1.txt", "w", encoding="utf-8") as file:
        with open("实验结果_剪枝前(体育15000语料_30).txt", "w", encoding="utf-8") as file:
            for phrase in sorted(hot_phrases.keys()):
                support, sources = hot_phrases[phrase]
                # if support > 1:
                line = f"频繁模式: {phrase}, 支持度: {support}, 共情来源: {sorted(sources)}\n"
                # print(line.strip())  # 打印到控制台
                file.write(line)  # 写入文件


class SubCorpusProcessor:
    def __init__(self, original_sentences, treehot, n1, n2):
        """
        初始化子语料处理器
        :param original_sentences: 原始完整语料列表（未经清洗）
        :param treehot: HotPhraseExtractor 实例，包含已构建的森林
        :param n1: 支持度阈值 n1
        :param n2: 过滤阈值 n2
        """
        self.original_sentences = original_sentences
        self.treehot = treehot
        self.n1 = n1
        self.n2 = n2
        self.global_hot_words = {}  # 存储全局频繁模式

    def process_sub_corpora(self):
        """主处理流程：对每个根节点的来源构建子邻接表并提取"""
        # 遍历森林中的每棵树的根节点
        for root in self.treehot.roots:
            # 获取该根节点对应的来源句子索引
            sources = root.sources
            # print(sources)
            # 根据来源索引提取对应的原始句子
            sub_sentences = [self.original_sentences[idx] for idx in set(sources)]
            # print(sub_sentences)

            # 构建子邻接表
            sub_adjacency = self._build_sub_adjacency(sub_sentences)

            # 提取子频繁模式
            sub_hot_words = self._extract_sub_hot_words(sub_adjacency)
            # print(sub_hot_words)

            #优化合并
            sub_hot_words=self._Phrase_sub_Merger(sub_hot_words,self.treehot.adjacency_list)

            # 构建全局森林并优化
            final_hot_phrases = self._build_global_forest(sub_hot_words)

            # 在合并到全局集合之前处理 final_hot_phrases 的 sources
            adjusted_hot_phrases = self._adjust_sources(final_hot_phrases, sources)
            # self._display_final_results(adjusted_hot_phrases)
            # 合并到全局集合
            self._merge_sub_results(adjusted_hot_phrases)
        self.global_hot_words=self._Phrase_sub_Merger(self.global_hot_words,self.treehot.adjacency_list)
        self._display_final_results(self.global_hot_words)

    def _adjust_sources(self, hot_phrases, original_sources):
        """调整热点短语的来源索引，以匹配原始语料中的索引"""
        # 将 original_sources 从 set 转换为 list
        original_sources_list = list(original_sources)
        adjusted_hot_phrases = {}
        for phrase, (support, sub_sources) in hot_phrases.items():
            # 将子来源索引映射到原始句子索引
            adjusted_sources = [original_sources_list[idx] for idx in sub_sources]
            adjusted_hot_phrases[phrase] = (support, adjusted_sources)
        return adjusted_hot_phrases

    def _build_sub_adjacency(self, sentences):
        """构建子邻接表"""
        adjacency = AdjacencyList()
        adjacency.process_and_build(sentences)
        return adjacency

    def _extract_sub_hot_words(self, adjacency):
        """从子邻接表提取频繁模式"""
        global_hot_words={}
        extractor = HotWordsExtractor(adjacency)
        extractor.extract(self.n1, mode=0)  # 使用默认模式
        extractor.filter_hot_words(self.n2)
        extractor.merge_hot_words(extractor.hot_words,global_hot_words)
        extractor.extract(self.n1, mode=1)  # 使用默认模式
        extractor.filter_hot_words(self.n2)
        extractor.merge_hot_words(extractor.hot_words, global_hot_words)
        return global_hot_words

    def _Phrase_sub_Merger(self,hotword,adjacency_list):
        """合并优化热词"""
        phrasemerger2 = PhraseMerger(hotword,adjacency_list)
        phrasemerger2.merge_included_phrases()
        # phrasemerger1.display_hot_words()      ####调试输出222
        phrasemerger2.merge_highly_similar_sources()
        phrasemerger2.display_hot_words()
        return phrasemerger2.global_hot_words

    def _merge_sub_results(self, hotwords):
        """合并子结果到全局，处理冲突"""
        for phrase, (support, sources) in hotwords.items():
            if phrase in self.global_hot_words:
                # 获取现有短语的支持度和来源
                existing_support, existing_sources = self.global_hot_words[phrase]
                # 计算新的支持度
                new_support = len(set(existing_sources) | set(sources))
                # 合并来源
                merged_sources = list(set(existing_sources) | set(sources))
                # 更新全局热点词
                self.global_hot_words[phrase] = (new_support, merged_sources)
            else:
                # 如果短语不存在，直接添加
                self.global_hot_words[phrase] = (support, sources)

    def _build_global_forest(self,hotwords):
        """构建全局森林并优化提取"""
        # 合并包含和相似短语
        # 构建优化森林
        tree = HotPhraseExtractor(hotwords, self.treehot.adjacency_list)
        tree.build_tree()
        tree.check_and_adjust_bottom_layer()
        tree.remove_weak_nodes_by_threshold(self.n2)
        return tree.extract_hot_phrases()

    def _display_final_results(self, hot_phrases):
        """展示最终热点"""
        # print("\n最终全局热点短语:")
        # for phrase, (support, sources) in sorted(hot_phrases.items(), key=lambda x: -x[1][0]):
        #     print(f"频繁模式: {phrase}, 支持度: {support}, 共情来源: {sorted(sources)}")
        print("\n最终全局热点短语:")

        # 打开文件准备写入
        # with open("实验结果(一万语料)2025_2_20_2.txt", "w", encoding="utf-8") as file:
        with open("实验结果_最终(体育15000语料_30).txt", "w", encoding="utf-8") as file:
            for phrase in sorted(hot_phrases.keys()):
                support, sources = hot_phrases[phrase]
                # if support > 1:
                line = f"频繁模式: {phrase}, 支持度: {support}, 共情来源: {sorted(sources)}\n"
                # print(line.strip())  # 打印到控制台
                file.write(line)  # 写入文件
# 运行主程序
if __name__ == "__main__":
    main()
