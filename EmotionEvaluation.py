# -*- coding: utf-8 -*-
import os
import codecs
import re
import collections
from janome.tokenizer import Tokenizer
import numpy as np
import pandas as pd

#第一引数に分析したいテキストを含んだファイル名を入力
df = pd.read_csv("out.txt",delimiter='\t', header = None)

af = pd.DataFrame()

# cvs stands for "Contextual Valence Shifters"
RE_PARTICLES = u'[だとはでがはもならじゃちってんすあ]*'
RE_CVS = u"いまひとつもない|なくても?問題ない|わけに[はも]?いかない|わけに[はも]?いくまい|いまひとつない|ちょ?っとも?ない|なくても?大丈夫|今ひとつもない|訳にはいくまい|訳に[はも]?[行い]かない|そんなにない|ぜったいない|まったくない|すこしもない|いまいちない|ぜんぜんない|そもそもない|いけない|ゼッタイない|今ひとつない|今一つもない|行けない|あまりない|なくていい|なくても?OK|なくても?結構|少しもない|今一つない|今いちない|言えるない|いえるない|行かん|あかん|いかん|なくても?良い|てはだめ|[ちじ]ゃだめ|余りない|絶対ない|全くない|今一ない|全然ない|もんか|ものか|あるますん|ない|いない|思うない|思えるない|訳[がではもじゃ]*ない|わけ[がではもじゃ]?ない"
CVS_TABLE = {
    'suki': ['iya'],
    'ikari': ['yasu'],
    'kowa': ['yasu'],
    'yasu': ['ikari', 'takaburi', 'odoroki', 'haji', 'kowa'],
    'iya': ['yorokobi', 'suki'],
    'aware': ['suki', 'yorokobi', 'takaburi', 'odoroki', 'haji'],
    'takaburi': ['yasu', 'aware'],
    'odoroki': ['yasu', 'aware'],
    'haji': ['yasu', 'aware'],
    'yorokobi': ['iya']
}

# Compiling regular expression patterns
BRACKET = u'\[|\(|\（|\【|\{|\〈|\［|\｛|\＜|\｜|\|'
EMOTICON_CHARS = u'￣|◕|´|_|ﾟ|・|｀|\-|\^|\ |･|＾|ω|\`|＿|゜|∀|\/|Д|　|\~|д|T|▽|o|ー|\<|。|°|∇|；|ﾉ|\>|ε|\)|\(|≦|\;|\'|▼|⌒|\*|ノ|─|≧|ゝ|●|□|＜|＼|0|\.|○|━|＞|\||O|ｰ|\+|◎|｡|◇|艸|Ｔ|’|з|v|∩|x|┬|☆|＠|\,|\=|ヘ|ｪ|ェ|ｏ|△|／|ё|ロ|へ|０|\"|皿|．|3|つ|Å|、|σ|～|＝|U|\@|Θ|‘|u|c|┳|〃|ﾛ|ｴ|q|Ｏ|３|∪|ヽ|┏|エ|′|＋|〇|ρ|Ｕ|‐|A|┓|っ|ｖ|∧|曲|Ω|∂|■|､|\:|ˇ|p|i|ο|⊃|〓|Q|人|口|ι|Ａ|×|）|―|m|V|＊|ﾍ|\?|э|ｑ|（|，|P|┰|π|δ|ｗ|ｐ|★|I|┯|ｃ|≡|⊂|∋|L|炎|З|ｕ|ｍ|ｉ|⊥|◆|゛|w|益|一|│|о|ж|б|μ|Φ|Δ|→|ゞ|j|\\|\    |θ|ｘ|∈|∞|”|‥|¨|ﾞ|y|e|\]|8|凵|О|λ|メ|し|Ｌ|†|∵|←|〒|▲|\[|Y|\!|┛|с|υ|ν|Σ|Α|う|Ｉ|Ｃ|◯|∠|∨|↑|￥|♀|」|“|〆|ﾊ|n|l|d|b|X|ó|Ő|Å|癶|乂|工|ш|ч|х|н|Ч|Ц|Л|ψ|Ψ|Ο|Λ|Ι|ヮ|ム|ハ|テ|コ|す|ｙ|ｎ|ｌ|ｊ|Ｖ|Ｑ|√|≪|⊇|⊆|＄|″|♂|±|｜|ヾ|？|：|ﾝ|ｮ|f|\%|ò|å|冫|冖|丱|个|凸|┗|┼|ц|п|Ш|А|φ|τ|η|ζ|β|α|Γ|ン|ワ|ゥ|ぁ|ｚ|ｒ|ｋ|ｄ|ｂ|Ｘ|Ｐ|Ｈ|Ｄ|８|♪|≫|↓|＆|「|［|々|仝|!|ﾒ|ｼ|｣'
RE_EMOTICON = re.compile('(' + BRACKET + ')([' + EMOTICON_CHARS + ']{3,}).*')
RE_POS = re.compile(u'感動|フィラー')
RE_MIDAS = re.compile(u'^(?:て|ね)(?:え|ぇ)$')
RE_KII = re.compile(
    '^aware$|^haji$|^ikari$|^iya$|^kowa$|^odoroki$|^suki$|^takaburi$|^yasu$|^yorokobi$')
RE_VALANCE_POS = re.compile('yasu|yorokobi|suki')
RE_VALANCE_NEG = re.compile('iya|aware|ikari|kowa')
RE_VALANCE_NEU = re.compile('takaburi|odoroki|haji')
RE_ACTIVATION_A = re.compile('takaburi|odoroki|haji|ikari|kowa')
RE_ACTIVATION_D = re.compile('yasu|aware')
RE_ACTIVATION_N = re.compile('iya|yorokobi|suki')


class MLAsk:
    def __init__(self, mecab_arg=''):
        self.mecab =  Tokenizer()
        self._read_emodic()

    def _read_emodic(self):
        """ Load emotion dictionaries """

        self.emodic = {'emotem': {}, 'emotion': {}}
        workingdir = os.path.abspath(os.path.dirname(__file__)) + '/'

        # Reading dictionaries of syntactical indicator of emotiveness
        emotemy = []
        for emotem_class in emotemy:
            filepath = "%semotions/%s.txt" % (workingdir, emotem_class)
            phrases = [l.strip() for l in codecs.open(filepath, 'r', 'utf8')]
            self.emodic['emotem'][emotem_class] = phrases

        # Reading dictionaries of emotion8
        emotions = (
        'aware', 'haji', 'ikari', 'iya', 'suki', 'takaburi',
        'yasu', 'yorokobi')
        for emotion_class in emotions:
            filepath = "%semotions/%s.txt" % (workingdir, emotion_class)
            phrases = [l.strip() for l in codecs.open(filepath, 'r', 'utf8')]
            self.emodic['emotion'][emotion_class] = phrases

    def analyze(self, text):
        """ Detect emotion from text """
        # Normalizing
        text = self._normalize(text)
        # Lemmatization by MeCab
        lemmas = self._lexical_analysis(text)
        # Finding emoticon
        emoticon = self._find_emoticon(text)
        # Finding intensifiers of emotiveness
        intensifier = self._find_emotem(lemmas, emoticon)
        intension = len(intensifier.values())
        # Finding words of emotion
        emotions = self._find_emotion(lemmas['all'])
        # Estimating sentiment orientation (POSITIVE, NEUTRAL, NEGATIVE)
        orientation = self._estimate_sentiment_orientation(emotions)
        # Estimating activeness (ACTIVE, NEUTRAL, PASSIVE)
        activation = self._estimate_activation(emotions)
        # Emotions display
        if emotions:
            result = {
                'text': text,
                'emotion': emotions,
                'orientation': orientation,
                'activation': activation,
                'emoticon': emoticon if emoticon else None,
                'intension': intension,
                'intensifier': intensifier,
                'representative': self._get_representative_emotion(emotions)
            }
        else:
            result = {
                'text': text,
                'emotion': None
            }
        return result

    def _normalize(self, text):
        text = text.replace('!', u'！').replace('?', u'？')
        return text


#形態素解析表示
    def _lexical_analysis(self, text):
        #print("analyze")
        """ By MeCab, doing lemmatisation and finding emotive indicator """
        lemmas = {'all': [], 'interjections': [], 'no_emotem': []}


        node1 = self.mecab.tokenize(text)
        for node in node1:
            print(node)
            surface = node.surface
            features=node.part_of_speech.split(",")
            pos, subpos, genkei = features[0], features[1], node.base_form
            lemmas['all'].append(genkei)
            if RE_POS.search(pos + subpos) or RE_MIDAS.search(surface):
                lemmas['interjections'].append(surface)
            else:
                lemmas['no_emotem'].append(surface)

        lemmas['all'] = ''.join(lemmas['all']).replace('*', '')
        lemmas['no_emotem'] = ''.join(lemmas['no_emotem'])
        #print(lemmas)
        return lemmas

    def _find_emoticon(self, text):
        """ Finding emoticon """
        emoticons = []
        if RE_EMOTICON.search(text):
            emoticon = RE_EMOTICON.search(text).group(1) + RE_EMOTICON.search(
                text).group(2)
            emoticons.append(emoticon)
        return emoticons

    def _find_emotem(self, lemmas, emoticon):
        """ Finding syntactical indicator of emotiveness """
        emotemy = {}
        for emotem_class, emotem_items in self.emodic['emotem'].items():
            found = []
            for emotem_item in emotem_items:
                if emotem_item in lemmas['no_emotem']:
                    found.append(emotem_item)
            if emotem_class == 'emotikony':
                if len(emoticon) > 0:
                    found.append(','.join(emoticon))
            elif emotem_class == 'interjections':
                if len(lemmas['interjections']) > 0:
                    found.append(''.join(lemmas['interjections']))

            if len(found) > 0:
                found = filter(lambda x: len(x) > 0, found)
                emotemy[emotem_class] = found
        return emotemy

    def _find_emotion(self, text):
        """ Finding emotion word by dictionaries """
        found_emotions = collections.defaultdict(list)
        for emotion_class, emotions in self.emodic['emotion'].items():
            for emotion in emotions:
                if emotion in text:
                    cvs_regex = re.compile(
                        '%s(?:%s(%s))' % (emotion, RE_PARTICLES, RE_CVS))
                    # if there is Contextual Valence Shifters
                    if cvs_regex.findall(text):
                        for new_emotion_class in CVS_TABLE[emotion_class]:
                            found_emotions[new_emotion_class].append(
                                emotion + "*CVS")
                    else:
                        found_emotions[emotion_class].append(emotion)
        return found_emotions if found_emotions else None

    def _estimate_sentiment_orientation(self, emotions):
        """ Estimating sentiment orientation (POSITIVE, NEUTRAL, NEGATIVE) """
        orientation = ''
        if emotions:
            how_many_valence = ''.join(emotions.keys())
            how_many_valence = RE_VALANCE_POS.sub('P', how_many_valence)
            how_many_valence = RE_VALANCE_NEG.sub('N', how_many_valence)
            how_many_valence = RE_VALANCE_NEU.sub('NorP', how_many_valence)
            num_positive = how_many_valence.count('P')
            num_negative = how_many_valence.count('N')
            if num_negative == num_positive:
                orientation = 'NEUTRAL'
            else:
                if num_negative > 0 and num_positive > 0:
                    orientation += 'mostly_'
                orientation += 'POSITIVE' if num_positive > num_negative else 'NEGATIVE'
            return orientation

    def _estimate_activation(self, emotions):
        """ Estimating activeness (ACTIVE, NEUTRAL, PASSIVE) """
        activation = ''
        if emotions:
            how_many_activation = ''.join(emotions.keys())
            how_many_activation = RE_ACTIVATION_A.sub('A', how_many_activation)
            how_many_activation = RE_ACTIVATION_D.sub('P', how_many_activation)
            how_many_activation = RE_ACTIVATION_N.sub('NEUTRAL',
                                                      how_many_activation)
            cnt_activation_A = how_many_activation.count('A')
            cnt_activation_P = how_many_activation.count('P')

            if cnt_activation_A == cnt_activation_P:
                activation = 'NEUTRAL'
            else:
                if cnt_activation_A > 0 and cnt_activation_P > 0:
                    activation = 'mostly_'
                activation += 'ACTIVE' if cnt_activation_A > cnt_activation_P else 'PASSIVE'
            return activation

    def _get_representative_emotion(self, emotions):
        '''
        感情語リストのうち一番長い語を代表的な感情として取り出す
        '''
        return \
        sorted(emotions.items(), key=lambda x: len(x[1][0]), reverse=True)[0]


""" Following codes are for debug """

def show_emowords(emo_hash):
    print
    '\n\t'.join([k + '-' + ','.join(v) for k, v in emo_hash.items() if k])
EMOTIONS_POINT = {'haji' : -1, 'ikari' : -1, 'iya': -1, 'aware': -1, 'suki': 1, 'takaburi': 1, 'yasu': 1, 'yorokobi': 1, 'odoroki': 0, 'kowa': 0}

def Evaluate(tweet):
    total = 0
    for i in range(len(list(result['emotion'].items()))):
        emotion = list(result['emotion'].items())  # どの感情か
        #print(emotion[i][0])
        #print(EMOTIONS_POINT[emotion[i][0]] * len(emotion[i][1]))
        total += EMOTIONS_POINT[emotion[i][0]] * len(emotion[i][1])
    return total

if __name__ == '__main__':

    userinput = df.ix[:, 0][0]  # 分析する言葉
    mlask = MLAsk()
    result = mlask.analyze(userinput)
    if result['emotion'] is not None:
        print(userinput)
        print(result['emotion'])
        print(Evaluate(result))
    else:
         print('Text:', result['text'])
         print('Emotion:', result['emotion'])

    """
    a = []
    b = []
    for i in range(len(df.ix[:,5])):
        #tweet_date = dt.strptime(df.ix[:, 2][i], '%Y-%m-%d %H:%M:%S')

        userinput = df.ix[:,5][i]   #分析する言葉
        mlask = MLAsk()
        result = mlask.analyze(userinput)
        if result['emotion'] is not None:
            #print(result['emotion'])

            #df['Point'] = [Evaluate(result)]
            #print(len(userinput))
            a.append(Evaluate(userinput))
            b.append(df.ix[:,2][i])
            #print(df.ix[:, 2][i])
            #df['Point'] = np.array()

        else:
            a.append(0)
            b.append(df.ix[:,2][i])
            #print('Text:', result['text'])
            #print('Emotion:', result['emotion'])
    af['Date'] = b
    af['Point'] = a
    af.to_csv("PointAdded.txt", sep = '\t', encoding='utf-8',index=False)

    print(af.head())
    """