# coding:utf-8

using DataFrames
using DataStructures

doc=
"""
dialogue act-tagging 
by using https://github.com/sanjaymeena/simplified_switchboard_dialog_act_corpus
"""

# bow
function to_bow(text, sep)
  text = [replace(l, r"\.|,","") for l in Vector(text)]
  segmentation = [split(i, sep) for i in text]
  words = unique(split(join(text, " "), sep))
  n, d = length(words), [ w => i for (i,w) in enumerate(words)]
  bow = zeros(length(text), n)
  for (i,seg) in enumerate(segmentation)
    i % 100 == 0 && println("i=$i")
    for s in seg
      !haskey(d, s) && continue
      bow[i, d[s]] += 1
    end
  end
  return bow, words
end

# swbd
df = readtable("switchboard_complete.csv", separator = ',', header = true)
df = df[!isna(df[:clean_text]), :]

# bow
bow, words = to_bow(df[:clean_text], r"\s+|,\s+")

# da
# act_label_1
# act_label_2

using DecisionTree

labels = Vector(df[:act_label_1])

model = build_forest(labels, bow, 2, 10, 0.5, 6)

accuracy = nfoldCV_forest(labels, bow, 2, 10, 3, 0.5)

@show accuracy

