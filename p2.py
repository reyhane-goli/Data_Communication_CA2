import math
import numpy as np
#************************NOISE FUNCTION**********************************************************
def noise(input_n):
    s=list(input_n)
    for i in range(len(s)):
        r = np.random.rand()
        if r < 0.1:
           diff = 1 - int(s[i])
           if(diff==1):
            s[i]='1'
           if(diff==0):
            s[i]='0'
    output ="".join(s)
    return output
#************************HUFFMAN FUNCTIONS*******************************************************
class HuffTree(object):
    def __init__(self, weight, symbol=None, zero=None, one=None):
        self.weight = weight
        self.symbol = symbol
        self.zero = zero
        self.one = one
    
def combine(tree1, tree2):
    return HuffTree(weight = tree1.weight+tree2.weight, zero=tree1, one=tree2)

def weight_dict_to_tree_nodes(weight_dict):
    nodes = []
    for w in weight_dict:
        nodes.append(HuffTree(weight_dict[w],w))
    return nodes


def make_huffman_tree(weight_dict):
    nodes = weight_dict_to_tree_nodes(weight_dict)
    return build(nodes)


def build(nodes):
    if len(nodes) == 1:
        return nodes[0]
    else:
        nodes.sort(key = lambda x: x.weight, reverse=True)
        n1 = nodes.pop()
        n2 = nodes.pop()
        nodes.append(combine(n2, n1))
        return build(nodes)

def huffman_coding(tree, codeword="", code_dict={}):

    if tree.symbol:
        if codeword == "":
            codeword = "0"
        code_dict[tree.symbol] = codeword
    else:
        huffman_coding(tree.zero, codeword+"0", code_dict)
        huffman_coding(tree.one, codeword+"1", code_dict)

    return code_dict


def return_huffman_code(user_message,code_dict):
    message_coded=""
    for i in user_message:
        temp=code_dict[i]
        message_coded=message_coded+str(temp)
    return message_coded


def huffman_decoding(code_message,tree):
    decoded=[]
    t = tree
    for i in code_message:
        if i == "0":
            t = t.zero
        elif i == "1":
            t = t.one
        else:
            raise Exception("Code_message not binary...Please try again")

        if t.symbol:
            decoded.append(t.symbol)
            t = tree

    return "".join(decoded)

weight_dict={
    'a':0.0816700000000000,
    'b':0.0149200000000000,
    'c':0.0278200000000000,
    'd':0.0425300000000000,
    'e':0.127020000000000,
    'f':0.0222800000000000,
    'g':0.0201500000000000,
    'h':0.0609400000000000,
    'i':0.0696600000000000,
    'j':0.00153000000000000,
    'k':0.00772000000000000,
    'l':0.0402500000000000,
    'm':0.0240600000000000,
    'n':0.0674900000000000,
    'o':0.0750700000000000,
    'p':0.0192900000000000,
    'q':0.000950000000000000,
    'r':0.0598700000000000,
    's':0.0632700000000000,
    't':0.0905600000000000,
    'u':0.0275800000000000,
    'v':0.00978000000000000,
    'w':0.0236000000000000,
    'x':0.00150000000000000,
    'y':0.0194700000000000,
    'z':0.00102000000000000
}
#**************************************END OF HUFFMAN FUNCTIONS*********************************************

#*************************************CONVOLUTIONAL ENCODER FUNCTIONS****************************************

conv_state_machine = {
    'zero': {'0': 
                {'parity':"00",'next_st': 'zero'},
             '1':
                {'parity':"11",'next_st': 'two'}
            },

    'one':  {'0': 
                {'parity': "10", 'next_st': 'zero'},
             '1':
                {'parity': "01", 'next_st': 'two'}
            },

    'two':  {'0': 
                {'parity': "11", 'next_st': 'one'},
             '1':
                {'parity': "00", 'next_st': 'three'}
            },

    'three':{'0':
                {'parity': "01", 'next_st': 'one'},
             '1':
                {'parity': "10", 'next_st': 'three'}
            },
}

def Convolutional_encode_st(message,conv_state_machine):
    conv_encoded_m=""
    prev_state='zero'
    for i in range(0,len(message)):
        one_or_zero=message[i]
        next_state=conv_state_machine[prev_state][one_or_zero]['next_st']

        
        conv_encoded_m=conv_encoded_m+conv_state_machine[prev_state][one_or_zero]['parity']
        prev_state=next_state
    return(conv_encoded_m)

#**************************************END OF CONVOLUTIONAL ENCODER FUNCTIONS*****************************

#**************************************VITERBI DECODER FUNCTIONS******************************************
start_PM_metric = {'zero':0,'one':math.inf, 'two': math.inf,'three':math.inf}
state_machine = {
    'zero': {'b1': 
                {'out_b':"00",'prev_st': 'zero','input_b':0},
             'b2':
                {'out_b':"10",'prev_st': 'one','input_b':0}
            },

    'one':  {'b1': 
                {'out_b': "11", 'prev_st': 'two', 'input_b': 0},
             'b2':
                {'out_b': "01", 'prev_st': 'three', 'input_b': 0}
            },

    'two':  {'b1': 
                {'out_b': "11", 'prev_st': 'zero', 'input_b': 1},
             'b2':
                {'out_b': "01", 'prev_st': 'one', 'input_b': 1}
            },

    'three':{'b1':
                {'out_b': "00", 'prev_st': 'two', 'input_b': 1},
             'b2':
                {'out_b': "10", 'prev_st': 'three', 'input_b': 1}
            },
}
 
def bits_diff_num(num_1,num_2):
    count=0;
    for i in range(0,len(num_1),1):
        if num_1[i]!=num_2[i]:
            count=count+1
    return count
 
def viterbi(encoded_message, start_PM_metric, state_machine):
    v_out_array=[None]*2
    PM = [{}]
    input_m=""
    correct_m=""
    for st in state_machine:
        PM[0][st] = {"metric": start_PM_metric[st]}
    PM.append({})
    for t in range(1, len(encoded_message)+1):
        for st in state_machine:
            prev_st = state_machine[st]['b1']['prev_st']
            first_b_metric = PM[0][prev_st]["metric"] + bits_diff_num(state_machine[st]['b1']['out_b'], encoded_message[t - 1])
            prev_st = state_machine[st]['b2']['prev_st']
            second_b_metric = PM[0][prev_st]["metric"] + bits_diff_num(state_machine[st]['b2']['out_b'], encoded_message[t - 1])
            if first_b_metric >= second_b_metric:
                PM[1][st] = {"metric" : second_b_metric,"branch":'b2'}
            else:
                PM[1][st] = {"metric": first_b_metric, "branch": 'b1'}

        min_val=math.inf
        for st in state_machine:
            if PM[1][st]["metric"]<=min_val:
                min_val=PM[1][st]["metric"]
                min_val_st=st

        last_branch=PM[1][min_val_st]["branch"]
        input_m=input_m+str(state_machine[min_val_st][last_branch]['input_b'])
        correct_m=correct_m+str(state_machine[min_val_st][last_branch]['out_b'])
        v_out_array[0]=input_m
        v_out_array[1]=correct_m
        for st in state_machine:
            PM[0][st]=PM[1][st]

    return (v_out_array)
#**************************************END OF VITERBI DECODER FUNCTIONS************************************

#***************************************DRIVE CODE FOR HUFFMAN*********************************************
tree = make_huffman_tree(weight_dict)
code_dict = huffman_coding(tree)
my_name="reyhanegoli"
M_coded= return_huffman_code(my_name,code_dict)
print("HUFFMAN ENCODE PART:my name coded is:",M_coded)
print("\n")
M_decoded =  huffman_decoding(M_coded,tree)
print("HUFFMAN DECODE PART:my name decoded:",M_decoded)
print("\n")
#***************************************END OF DRIVE CODE FOR HUFFMAN************************************

#***************************************DRIVE CODE FOR CONV***********************************************
conv_M=Convolutional_encode_st(M_coded,conv_state_machine)
print("CONVOLUTIONAL ENCODE PART:convolutional encode is:",conv_M)
print("\n")
#**************************************END OF DRIVE CODE FOR CONV****************************************** 

conv_M_noisey=noise(conv_M)
print("NOISE PART: Message after noise function is:",conv_M_noisey)
print("\n")

#**************************************DRIVE CODE FOR VETERBI**********************************************
j=0
c=""
size_arr=int(len(conv_M_noisey)/2)
arr=[None]*size_arr
for i in range(0,len(conv_M_noisey),2):
    arr[j]=c+str(conv_M_noisey[i])+str(conv_M_noisey[i+1])
    j=j+1
viterbi_out_array=[]
viterbi_out_array=viterbi(arr,start_PM_metric,state_machine)
print("VITERBI DECODE PART:input message is:",viterbi_out_array[0])
print("\n")
print("VITERBI DECODE PART:correct message is:",viterbi_out_array[1])
print("\n")

#**************************************END OF DRIVE CODE FOR VETERBI***************************************

M_decoded_noise =  huffman_decoding(viterbi_out_array[0],tree)
print("HUFFMAN DECODE AFTER NOISE FUNCTION:my name decoded after noise func:",M_decoded_noise)