# -*- coding: utf-8 -*-

pad = '<pad>'
unk = '<unk>'
bos = '<bos>'
eos = '<eos>'
nul = '<nul>'

pos_label = ["NN", "VV", "PU", "AD", "NR", "PN", "P", "CD", "M", "VA", "DEG", "JJ", "DEC", "VC", "NT", "SP", "DT", "LC",
             "CC", "AS", "VE", "IJ", "OD", "CS", "MSP", "BA", "DEV", "SB", "ETC", "DER", "LB", "IC", "NOI", "URL", "EM",
             "ON", "FW", "X"]

onto4_label = ['<pad>', 'O', 'ORG', 'GPE', 'LOC', 'PER']    # 5
onto5_label = ['<pad>', 'O', 'ORG', 'DATE', 'MONEY', 'PERSON', 
               'GPE', 'CARDINAL', 'ORDINAL', 'PERCENT', 
               'LOC', 'PRODUCT', 'LAW', 'LANGUAGE', 'EVENT', 
               'NORP', 'TIME', 'WORK_OF_ART', 'FAC', 'QUANTITY']   # 19
