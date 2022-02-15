from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

def bin_class_report(X_test,y_test, model):
    """
    modified classification report for binary output
    """
    NLL, Acc = model.evaluate( x=X_test, y=y_test, verbose=0)
    y_pred = model.predict(X_test.reshape(len(X_test), 192,192,1))
    
    # cm , AUC
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis =1))
    AUC =  metrics.roc_auc_score(y_test.argmax(axis=1), y_pred.argmax(axis =1))
    
    #acc
    nobs = sum(sum(cm))
    count = sum([cm[0,0], cm[1,1]])
    acc_ci_low, acc_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    
    #sens 
    sens = cm[1,1]/(cm[1,1]+cm[1,0])
    nobs = sum([cm[1,0],cm[1,1]])
    count = sum([cm[1,1]])
    sens_ci_low, sens_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    
    #spec 
    spec = cm[0,0]/(cm[0,1]+cm[0,0])
    nobs = sum([cm[0,1],cm[0,0]])
    count = sum([cm[0,0]])
    spec_ci_low, spec_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    
    print("\nPerformance on Test Set : ")
    print("\nAccuracy    [95% Conf.] :", np.around(Acc,4),np.around([acc_ci_low, acc_ci_upp],4))
    print("Sensitivity [95% Conf.] :", np.around(sens,4), np.around([sens_ci_low, sens_ci_upp],4))
    print("Specificity [95% Conf.] :", np.around(spec,4), np.around([spec_ci_low, spec_ci_upp],4))
    print("\nArea under Curve (AUC)  :", np.around(AUC,4))
    print("Negative Log-Likelihood :", np.around(NLL, 4))
    #print(metrics.classification_report(y_test.argmax(axis=1), y_pred.argmax(axis =1)))

    