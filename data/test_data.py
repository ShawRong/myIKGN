import pickle
import pandas as pd

file_names = ['nyc_category.pkl', 'tky_category.pkl']
file_name = file_names[0]

dataset = None
with open(file_name, 'rb') as file:
    dataset = pickle.load(file)

print(f"type: {type(dataset)}")
print("Keys:", dataset.keys())
print("Number of items:", len(dataset))
n = 10
for key in dataset.keys():
    print(f"type: {type(dataset[key])}")


dict_key = ['data_neural', 'vid_list', 'uid_list', 'parameters', 'data_filter', 'vid_lookup', 'KG']
list_key = ['category_name']
for key1 in dict_key:
    print(f"KEY:{key1}: ")
    for key2 in list(dataset[key1].keys())[:10]:
        print(f"\t key:{key2}: value type: {type(dataset[key1][key2])}") 

print(dataset[list_key[0]][:10])

print(dataset['data_neural'][0].keys())
for key in dataset['data_neural'][0].keys():
    print(f"key{key}, val type: {type(dataset['data_neural'][0][key])}")
#print("session data:", dataset['data_neural'][0]['sessions'])
#print("session data trans:", dataset['data_neural'][0]['sessions_trans'])
print("session data trans:", dataset['data_neural'][0]['train'])
print("session data trans:", dataset['data_neural'][0]['test'])
print("session data trans:", dataset['data_neural'][0]['vaild'])
#print('category name', dataset['category_name'])
#print('vid list', dataset['vid_list'])
#print('uid list', dataset['uid_list'])
print('parameters', dataset['parameters'])
#print('data filters', dataset['data_filter']['1'])
#print('vid lookup', dataset['vid_lookup'])
print('kg', dataset['KG'].keys())
#keys = ['utp', 'ptp', 'ptp_dict', 'poi_trans', 'timining_rel', 'tim_rel', 'dis_rel', 'train_kg', 'max_dis_tim']
#keys_todo = ['utp', 'ptp', 'ptp_dict', 'poi_trans', 'timining_rel', 'tim_rel', 'dis_rel', 'train_kg', 'train_kg_dict', 'max_dis_tim']
#for key in keys:
    #print(dataset['KG'][key])
#print('tim_rel', dataset['KG']['tim_rel'])
#print('dis_rel', dataset['KG']['dis_rel'])

print('tim_max', dataset['KG']['tim_max'])
print('dis_max', dataset['KG']['dis_max'])
print('timining_rel', dataset['KG']['timining_rel'])


""" Note:
dataset:
Keys: ['data_neural', 'vid_list', 'uid_list', 'category_name', 'parameters', 'data_filter', 'vid_lookup', 'KG']
is dict: ['data_neural', 'vid_list', 'uid_list', 'parameters', 'data_filter', 'vid_lookup', 'KG']
is list: ['category_name']


data_neural: data used to train the network.
    containing keys: [sessions, train, test, (vaild)(ATTENTION: CODER TYPO ERROR), sessions_trans]
    dict: [sessions, sessions_trans]
        sessions:       key: 0, 1, ... 136, val: [11, 2], [3, 4], ...
        sessions_trans: key: 0, 1, ... 136, val: [49, (0, 2), 46], ...
    list: [train, test, vaild]
        session data trans: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ...]
        session data trans: [123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]
        session data trans: [109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]
        
vid_list: venue list. what the coder used is just len(vid_list)
    {'4b5f42f1f964a520a2b029e3': [13727, 3, '4bf58dd8d48988d147941735'], ...}
uid_list: user list. what the coder used is just len(uid_list)
    {'426': [1018, 10], '1081': [1019, 14], ...}
parameters: not used in code, at least I didn't see it. I suppose it's some parameter checkpoint of network.
    parameters {'TWITTER_PATH': 'D:\\STKGRec-main\\data\\dataset_TSMC2014_NYC.txt',
    'SAVE_PATH': 'D:\\STKGRec-main\\data\\', 'trace_len_min': 10,
    'location_global_visit_min': 10,
    'hour_gap': 24, 'min_gap': 10,
    'session_max': 10, 'filter_short_session': 3,
    'sessions_min': 5, 'train_split': 0.8}
data_filter: not used, too.
    {{'sessions_count': 7, 'topk_count': 81, 'topk': [('49d2b43ef964a520cb5b1fe3', 7), ('4d4ac10da0ef54814b6ffff6', 4), ('42accc80f964a52047251fe3', 4), ('4fbf92b7e4b08821682bf100', 4), ('42586c80f964a520db201fe3', 3),}...}
vid_lookup: used in main.py line 297 & line 307
    {..., 14084: [-73.79681911801289, 40.72192402665647]}
KG: 
    'utp', 'ptp', 'ptp_dict', 'poi_trans', 'timining_rel', 'tim_rel', 'dis_rel', 'train_kg', 'train_kg_dict', 'max_dis_tim'
    tim_rel: time gap? 
        tim_rel [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -25, -24, -23, -22, -21, -20, -19, -17, -15]

        tim_rel (Time Relations)
        This represents discrete time intervals or gaps between POI visits
        Range: [-66 to 24]
        Structure:
        Positive values [0-24]: Likely represent hours in a day
        Negative values [-66 to -15]: Likely represent historical time gaps in reverse order
        Purpose: Used to model temporal relationships between POIs in the knowledge graph

    dis_rel: distance gap? 
        dis_rel [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 471, 473, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 527, 528, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 544, 546, 547, 548, 550, 552, 553, 556, 558, 560, 561, 563, 565, 566, 567, 569, 570, 571, 572, 576, 577, 579, 582, 584, 585, 586, 587, 589, 590, 591, 597, 598, 601, 602, 603, 604, 608, 610, 616, 618, 620, 626, 627, 628, 630, 631, 633, 640, 642, 644, 645, 649, 651, 653, 654, 661, 662, 670, 671, 673, 675, 695, 700, 717, 724, 732, 733, 734, 735, 736, 737, 738, 778, 817, 827, 829, 840]
        Represents discretized distance intervals between POIs
        Range: [0-840]
        Structure:
        Continuous sequence of integers representing distance bins
        Higher values indicate greater distances between POIs
        Purpose: Used to model spatial relationships between POIs in the knowledge graph

    timining_rel: timing sequence relation

    poi_trains: the transformation relation between POIs.
    ptp_dict: POI to POI mapping dict.
    traini_kg: triplet for training.

category_name: used only to get the number of category.
    ['4bf58dd8d48988d1bc941735', '4bf58dd8d48988d101941735', ...]

KEY:data_neural:
         key:0: value type: <class 'dict'>
         key:1: value type: <class 'dict'>
         etc ...

KEY:vid_list:
         key:unk: value type: <class 'list'>
         key:4ae8fd76f964a520e1b321e3: value type: <class 'list'>
         key:4b679336f964a520d9552be3: value type: <class 'list'>
         etc ...

KEY:uid_list:
         key:293: value type: <class 'list'>
         key:445: value type: <class 'list'>
         etc ...

KEY:parameters: omit

KEY:data_filter:
         key:768: value type: <class 'dict'>
         key:445: value type: <class 'dict'>
         etc ...

KEY:vid_lookup:
         key:9: value type: <class 'list'>
         key:10: value type: <class 'list'>
         etc ...

KEY:KG:
         key:utp: value type: <class 'dict'>
         key:ptp: value type: <class 'dict'>
         key:ptp_dict: value type: <class 'dict'>
         key:poi_trans: value type: <class 'dict'>
         key:timining_rel: value type: <class 'list'>
         key:tim_rel: value type: <class 'list'>
         key:dis_rel: value type: <class 'list'>
         key:train_kg: value type: <class 'numpy.ndarray'>
         key:train_kg_dict: value type: <class 'collections.defaultdict'>
         key:max_dis_tim: value type: <class 'list'>
         etc ...


'category_name' (list): omit
"""