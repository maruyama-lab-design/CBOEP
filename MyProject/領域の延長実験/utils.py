from random import randint


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
    sentence = []

    while len(sequence) >= k_min:
        now_k = randint(k_min, k_max)
       	sentence.append(sequence[:now_k])
        sequence = sequence[now_k:]
		
    return sentence
