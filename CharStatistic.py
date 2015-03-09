__author__ = 'v-penlu'
import xml.etree.ElementTree as ET
from collections import Counter
from random import shuffle


xml_file = r"D:\newsspace200.xml"
tree = ET.parse(xml_file)
root = tree.getroot()
triple = []
category_dic = dict()
news_list = []
for child in root:
    if child.tag == "title":
        if child.text is None:
            triple.append("")
        else:
            triple.append(child.text)
    elif child.tag == "description":
        if child.text is None:
            triple.append("")
        else:
            triple.append(child.text)
    elif child.tag == "category":
        if child.text is None:
            triple.append("None")
        else:
            triple.append(child.text)
    elif child.tag == "pubdate":
        if triple[1] not in category_dic:
            category_dic[triple[1]] = len(category_dic)
            news_list.append([])
            news_list[category_dic[triple[1]]].append([triple[0], triple[2]])
        else:
            news_list[category_dic[triple[1]]].append([triple[0], triple[2]])
        triple = []

f = open("news.txt", 'w')
for k, v in category_dic.items():
    print >> f, k
    for p in news_list[v]:
        print >> f, p[0], p[1]
    print >> f
f.close()

chars = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_stat = []
for v in range(len(category_dic)):
    char_stat.append([])
    for pair in news_list[v]:
        cnt = Counter()
        s = pair[0] + pair[1]
        s = s.lower()
        if s != "":
            for c in s:
                cnt[c] += 1
            stat = [cnt[x] for x in chars]
            char_stat[v].append([cnt[x] for x in chars])

four_categ = ["World", "Sports", "Business", "Sci/Tech"]

f_train = open("train_75.arff", "w")
f_test = open("test_75.arff", "w")
print >> f_train, "@RELATION news_classify\n"
print >> f_test, "@RELATION news_classify\n"
for i, c in enumerate(chars):
    print >> f_train, "@ATTRIBUTE prob_%d  NUMERIC" % i
    print >> f_test, "@ATTRIBUTE prob_%d NUMERIC" % i
class_str = ",".join(four_categ)
class_str = "{" + class_str + "}"
print >> f_train, "@ATTRIBUTE class %s\n\n@DATA" % class_str
print >> f_test, "@ATTRIBUTE class %s\n\n@DATA" % class_str

out_format = "%.6f," * len(chars) + "%s"
for k, v in category_dic.items():
    if k in four_categ:
        length = len(char_stat[v])
        idx = range(length)
        shuffle(idx)
        train_idx = idx[0:40000]
        test_idx = idx[40000:41100]
        for sample_idx in train_idx:
            # 0.75 refers to word2vec unigram table
            un_norm = map(lambda a: pow(float(a), 0.75), char_stat[v][sample_idx])
            total = sum(un_norm)
            prob = map(lambda a: a / total, un_norm)
            prob.append(k)
            print >> f_train, out_format % tuple(prob)
        for sample_idx in test_idx:
            un_norm = map(lambda a: pow(float(a), 0.75), char_stat[v][sample_idx])
            total = sum(un_norm)
            prob = map(lambda a: a / total, un_norm)
            prob.append(k)
            print >> f_test, out_format % tuple(prob)


