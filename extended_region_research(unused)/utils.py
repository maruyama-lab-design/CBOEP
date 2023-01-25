from random import randint
import pickle


def make_kmer_list(k, stride, sequence):
	#-----説明-----
	# sequence(塩基配列) を k-mer に区切り、sentence で返す
	# 返り値である sentence は k-mer(word) のリスト
	#-------------

	sequence = sequence.replace("\n", "") # 改行マークをとる
	length = len(sequence)
	sentence = []
	start_pos = 0
	while start_pos <= length - k:
		# k-merに切る
		word = sequence[start_pos : start_pos + k]
		
		# 切り出したk-merを書き込む
		sentence.append(word)

		start_pos += stride

	return sentence


def make_random_kmer_list(k_min, k_max, sequence):
	'''
	Split sequence to random length k-mer that has k_min ≦ k-mer ≦ k_max
	'''
	if len(sequence) < k_min: # 例外
		print(sequence)
		return [sequence]

	sentence = []
	start_i = 0
	while start_i < len(sequence):
		if len(sequence) - 2 * k_min < start_i:
			sentence.append(sequence[start_i:])
			break
		now_k = randint(k_min, min(k_max, len(sequence) - start_i - k_min)) # 配列外参照しないように
		sentence.append(sequence[start_i : start_i + now_k])
		start_i += now_k	
	return sentence


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


