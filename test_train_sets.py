def get_test_set(): # 30 in test set
    return ['pp71', 'p49', 'p52', 'p83', 'pp6', 
            'pp83', 'pp59', 'pp56', 'p48', 'pp89', 
            'p47', 'p3', 'pp21', 'pp15', 'pp49', 
            'pp33', 'pp62', 'pp72', 'pp64', 'p89', 
            'pp25', 'pp37', 'p1', 'p63', 'pp35', 
            'p37', 'pp78', 'p7', 'p10', 'pp58']

def get_train_set(num): # 27 
    if num == 0:
        return ['p12', 'pp84', 'p50', 'p59', 'pp24',
                'pp45', 'p55', 'p56', 'pp81', 'p72',
                'pp48', 'p27', 'pp70', 'pp8', 'pp53',
                'pp11', 'pp14', 'pp47', 'pp32', 'p79',
                'p11', 'p70', 'p4', 'pp5', 'p77', 
                'p33', 'pp43']
    elif num == 1:
        return ['p24', 'p86', 'pp22', 'p5', 'pp74', 
                'p58', 'p73', 'p76', 'pp4', 'pp30', 
                'p32', 'p81', 'pp57', 'pp13', 'p65', 
                'pp67', 'p14', 'pp50', 'pp77', 'p78', 
                'pp16', 'pp44', 'p35', 'pp1', 'p34', 
                'pp42', 'p84']
    elif num == 2:
        return ['p45', 'p43', 'p31', 'p15', 'p29', 
                'pp34', 'pp69', 'pp12', 'p30', 'pp55', 
                'p80', 'p61', 'pp31', 'pp79', 'pp76', 
                'p85', 'p17', 'pp86', 'pp27', 'pp52', 
                'p67', 'p53', 'p21', 'pp3', 'p71', 'p62', 
                'pp29']
    elif num == 3:
        return ['pp60', 'p16', 'pp63', 'p74', 'p44', 'p64', 
                'pp17', 'p25', 'pp10', 'p13', 'p22', 'p42', 
                'p20', 'pp61', 'p66', 'pp65', 'pp66', 'pp80', 
                'pp7', 'p57', 'pp85', 'pp73', 'p8', 'p6', 
                'pp20', 'p69', 'p60']
    else:
        print("Invalid training set selection, please input an integer 0-3")
        return None

