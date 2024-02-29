# Starter code for CS 165B MP1 Spring 2023

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """
    import numpy as np
    
    D = training_input[0][0]
    #print(training_input[1:])
    #training_input = np.matrix(training_input)
    N1,N2,N3 = training_input[0][1], training_input[0][2], training_input[0][3]
        
    train_input = training_input[1:]
    train_input = np.array(train_input)

    class1 = train_input[0:N1]
    class2 = train_input[N1:N1+N2]
    class3 = train_input[N1+N2:N1+N2+N3]

    Dte = testing_input[0][0]
    t1,t2,t3 = testing_input[0][1], testing_input[0][2], testing_input[0][3]
    
    test_input = testing_input[1:]
    test_input = np.array(test_input)
    
    #print(test_input.shape)
    #test1 = test_input[0:100]
    #test2 = test_input[100:200]
    #test3 = test_input[200:300]

    #c1 = np.matrix
    c1 = np.mean(class1,axis=0)
    c2 = np.mean(class2,axis=0)
    c3 = np.mean(class3,axis=0)
    
    #print(c1,c2,c3)
    
    #print(c1,c2,test_input[0])
    #print(disfunc(c1,c2,test_input[0]))
    #m1,m2,m3 = (c1+c2)/2.0, (c1+c3)/2.0, (c2+c3)/2.0
    
    tp,fp,fn,tn = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
    #print(disfunc(c1,c2,test_input[20]))

    tpr,fpr,err,acc,prec = np.zeros(3, dtype = 'float32'),np.zeros(3, dtype = 'float32'),np.zeros(3, dtype = 'float32'),np.zeros(3, dtype = 'float32'),np.zeros(3, dtype = 'float32')
    
    for i in range(0,t1):
        
        pred = disfunc(c1,c2,test_input[i]) # A vs B

        if pred <= 0: # A or C
            pred = disfunc(c1,c3,test_input[i])
            #print(c1,c3,test_input[i],pred1)
            tn[1] += 1
            if pred <= 0: # pos: A
                tp[0] += 1
                tn[2] += 1
            else: # C
                fn[0] += 1
                fp[2] += 1
        else: # B or C
            pred = disfunc(c2,c3,test_input[i])
            fn[0] += 1
            if pred <= 0: # B
                fp[1] += 1
                tn[2] += 1
            else: # C
                tn[1] += 1
                fp[2] += 1  
                
    #print(fp[0])
        
    for i in range(t1, t1+t2):
        pred = disfunc(c1,c2,test_input[i])
        if pred <= 0: # A or C
            pred = disfunc(c1,c3,test_input[i])
            fn[1] += 1
            if pred <= 0: # pos: A
                fp[0] += 1
                tn[2] += 1
            else: # C
                tn[0] += 1
                fp[2] += 1
        else: # B or C
            pred = disfunc(c2,c3,test_input[i])
            tn[0] += 1
            if pred <= 0:
                tp[1] += 1
                tn[2] += 1
            else:
                fn[1] += 1
                fp[2] += 1 
            
    for i in range(t1+t2, t1+t2+t3):
        pred = disfunc(c1,c2,test_input[i])
        if pred <= 0: # A or C
            pred = disfunc(c1,c3,test_input[i])
            tn[1] += 1
            if pred <= 0: # pos: A
                fp[0] += 1
                fn[2] += 1
            else: # C
                tn[0] += 1
                tp[2] += 1
        else: # B or C
            pred = disfunc(c2,c3,test_input[i])
            tn[0] += 1
            if pred <= 0:
                fp[1] += 1
                fn[2] += 1
            else:
                tn[1] += 1
                tp[2] += 1 
    
    for i in range(3):
        tpr[i] = tp[i] / (tp[i]+fn[i])
        fpr[i] = fp[i] / (fp[i]+tn[i])
        err[i] = (fp[i]+fn[i]) / (fp[i]+fn[i]+tp[i]+tn[i])
        acc[i] = (tp[i]+tn[i]) / (fp[i]+fn[i]+tp[i]+tn[i])
        prec[i] = tp[i] / (tp[i]+fp[i])
    print(tp[0],prec[0],tpr[0])
    
    mtpr = np.mean(tpr)
    mfpr = np.mean(fpr)
    merr = np.mean(err)
    macc = np.mean(acc)
    mprec = np.mean(prec)
    print(mtpr,mfpr,merr,macc,mprec)
    
    res = {
                "tpr": mtpr,
                "fpr": mfpr,
                "error_rate": merr,
                "accuracy": macc,
                "precision": mprec
    }
    
    return res
    

    #print(d)

    # TODO: IMPLEMENT
    #pass
    
def disfunc(cent1,cent2, x):
    # if disfunc<0, x in cent1.
    
    m = (cent1 + cent2) / 2.0

    cent = cent2 - cent1
    cent = cent.T
    
    x = x - m
    
    return cent@x
