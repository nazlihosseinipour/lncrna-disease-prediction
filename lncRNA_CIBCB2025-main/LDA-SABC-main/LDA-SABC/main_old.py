from CV import kfold

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve,auc
import pandas as pd
import numpy as np
import csv
import os
import datetime
now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("time1:", now1)


n_features = 64
lgbn_estimators = 100
CNNn_estimators = 100
lgblearning_rate = 0.1
CNNlearning_rate = 0.1
CNNepochs = 10
batch_size = 10


id = 0
time = 1
k = 5
alpha = 0.4
parameter = f'n_features{n_features},alpha{alpha},lgbn_estimators{lgbn_estimators},CNNn_estimators{CNNn_estimators},' \
            f'lgblearning_rate{lgblearning_rate},CNNlearning_rate{CNNlearning_rate},' \
            f'CNNepochs{CNNepochs},batch_size{batch_size},time{time}-1,k{k},linear'
for mm in range(1, 2):
    #for cv in range(1, 5):
    for cv in range(4, 5):
        data_file = './data' + str(mm) + '/data32linear.csv'
        label_file = './data' + str(mm) + '/label.csv'
        data = pd.read_csv(data_file, header=None, index_col=None).to_numpy()
        label = pd.read_csv(label_file, index_col=None, header=None).to_numpy()
        label_copy = label.copy()
        row, col = label.shape
        if cv == 4:
            c = np.array([(i, j) for i in range(row) for j in range(col)])
        else:
            a = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j]])
            b = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j] == 0])
            np.random.shuffle(b)
            sample = len(a)
            b = b[:sample]
        mPREs = np.array([])
        mACCs = np.array([])
        mRECs = np.array([])
        mAUCs = np.array([])
        mAUPRs = np.array([])
        mF1 = np.array([])

        for j in range(time):
            if cv == 4:
                c_tr, c_te = np.array(kfold(c, k=k, row=row, col=col, cv=cv))
            elif cv == 3:
                a_tr, a_te = np.array(kfold(a, k=k, row=row, col=col, cv=cv))
                b_tr, b_te = np.array(kfold(b, k=k, row=row, col=col, cv=cv))
            else:
                c = np.vstack([a, b])
                kf = kfold(c, k=k, row=row, col=col, cv=cv)
                ws_tr = []
                ws_te = []

                c_tr, c_te = np.array(kfold(c, k=k, row=row, col=col, cv=cv))
            for i in range(k):
                if cv == 4:
                    b_tr = []
                    a_tr = []
                    # print(c_tr[i])
                    for ep in c_tr[i]:
                        if label_copy[c[ep][0]][c[ep][1]]:
                            b_tr.append(c[ep])
                        else:
                            a_tr.append(c[ep])
                    b_te = []
                    a_te = []
                    for ep in c_te[i]:
                        if label_copy[c[ep][0]][c[ep][1]]:
                            b_te.append(c[ep])
                        else:
                            a_te.append(c[ep])
                    b_te = np.array(b_te)
                    b_tr = np.array(b_tr)
                    a_te = np.array(a_te)
                    a_tr = np.array(a_tr)
                    np.random.shuffle(b_te)
                    np.random.shuffle(a_te)
                    np.random.shuffle(b_tr)
                    np.random.shuffle(a_tr)
                    a_tr = a_tr[:len(b_tr)]
                    a_te = a_te[:len(b_te)]
                    train_sample = np.vstack([a_tr, b_tr])
                    test_sample = np.vstack([a_te, b_te])
                elif cv == 3:
                    train_sample = np.vstack([np.array(a[a_tr[i]]), np.array(b[b_tr[i]])])
                    test_sample = np.vstack([np.array(a[a_te[i]]), np.array(b[b_te[i]])])
                else:
                    train_sample = np.array(c[c_tr[i]])
                    test_sample = np.array(c[c_te[i]])
                train_land = train_sample[:, 0] * col + train_sample[:, 1]
                test_land = test_sample[:, 0] * col + test_sample[:, 1]
                np.random.shuffle(train_land)
                np.random.shuffle(test_land)
                X_tr = data[train_land][:, :-1]
                y_tr = data[train_land][:, -1]
                X_te = data[test_land][:, :-1]
                y_te = data[test_land][:, -1]
                #
                # model1 = BoostingMachine(objective='logloss',num_round= 1000,use_gpu =False,learning_rate=0.001,min_max_depth=1,max_max_depth=25,subsample=0.8)
                # model1.fit(X_tr, y_tr)
                #
                # score = model1.predict_proba(X_te)[:,1]

                import Code.test2_CNN as test2_CNN
                from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN
                import lightgbm as lgb

                id = id + 1

                model = lgb.LGBMClassifier(learning_rate=lgblearning_rate, n_estimators=lgbn_estimators)
                model.fit(X_tr, y_tr)
                lscore = model.predict_proba(X_te)
                lscore = lscore[:, 1]


                X_train_r = test2_CNN.reshape_for_CNN(X_tr)
                X_test_r = test2_CNN.reshape_for_CNN(X_te)
                bdt_real_test_CNN = Ada_CNN(base_estimator=test2_CNN.baseline_model(n_features=n_features),
                                            n_estimators=CNNn_estimators,
                                            learning_rate=CNNlearning_rate, epochs=CNNepochs)
                bdt_real_test_CNN.fit(X_train_r, y_tr, batch_size = batch_size)
                # cpre_label = bdt_real_test_CNN.predict(X_test_r)
                cscore = bdt_real_test_CNN.predict_proba(X_test_r)
                cscore = cscore[:, 1]

                score = alpha * cscore + (1-alpha) * lscore

                pred = []
                prob =  score
                for n in prob:
                    if n > 0.5:
                        pred.append(1)
                    else:
                        pred.append(0)
                pre_label = np.array(pred)

                fpr, tpr, threshold = roc_curve(y_te, score)
                pre, rec_, _ = precision_recall_curve(y_te, score)

                acc = accuracy_score(y_te, pre_label)
                rec = recall_score(y_te, pre_label)
                f1 = f1_score(y_te, pre_label)
                Pre = precision_score(y_te, pre_label)
                au = auc(fpr, tpr)
                apr = auc(rec_, pre)
                mPREs = np.append(mPREs, Pre)
                mACCs = np.append(mACCs, acc)
                mRECs = np.append(mRECs, rec)
                mAUCs = np.append(mAUCs, au)
                mAUPRs = np.append(mAUPRs, apr)
                mF1 = np.append(mF1, f1)

                curve_1 = np.vstack([fpr, tpr])
                curve_1 = pd.DataFrame(curve_1.T)
                curve_1.to_csv('./co_s/d' + str(mm) + 'c' + str(cv) + f'_time{time}_' + str(id) + '_c' + str(au) + '.csv',
                               header=None, index=None)

                curve_2 = np.vstack([rec_, pre])
                curve_2 = pd.DataFrame(curve_2.T)
                curve_2.to_csv('./co_s/d' + str(mm) + 'c' + str(cv) + f'_time{time}_' + str(id) + '_r' + str(apr) + '.csv',
                               header=None, index=None)

                print('Precision is :{}'.format(Pre))
                print('Recall is :{}'.format(rec))
                print("ACC is: {}".format(acc))
                print("F1 is: {}".format(f1))
                print("AUC is: {}".format(au))
                print('AUPR is :{}'.format(apr))

            print(f"{j+1} time 5-Folds computed.")

        toa = np.vstack([mPREs,mRECs,mACCs,mF1,mAUCs, mAUPRs])
        toa = pd.DataFrame(toa)
        if not os.path.exists(f'./res_data'):
            os.makedirs(f'./res_data')
        toa.to_csv('res_data/' + 'cv' + str(cv) + '_data' + str(mm) + str(parameter) + '.csv', header=None, index=None)
        PRE = mPREs.mean()
        REC = mRECs.mean()
        ACC = mACCs.mean()
        F1 = mF1.mean()
        AUC = mAUCs.mean()
        AUPR = mAUPRs.mean()

        PRE_err = np.std(mPREs)
        ACC_err = np.std(mACCs)
        REC_err = np.std(mRECs)
        AUC_err = np.std(mAUCs)
        AUPR_err = np.std(mAUPRs)
        F1_err = np.std(mF1)

        print('\n')
        print("PRE is:{}±{}".format(round(PRE, 4), round(PRE_err, 4)))
        print("REC is:{}±{}".format(round(REC, 4), round(REC_err, 4)))
        print("ACC is:{}±{}".format(round(ACC, 4), round(ACC_err, 4)))
        print("F1 is:{}±{}".format(round(F1, 4), round(F1_err, 4)))
        print('AUC is :{}±{}'.format(round(AUC, 4), round(AUC_err, 4)))
        print('AUPR is :{}±{}'.format(round(AUPR, 4), round(AUPR_err, 4)))
        now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("time2:", now2)

        f_1 = open('./co_s/' + 'res_cv.txt', 'a')
        f_1.write('data' + str(mm) + 'cv' + str(cv) + ':\n')
        f_1.write('now1' + str(now1) + '   ' + 'now2' + str(now2) + ':\n')
        f_1.write('n_features' + str(n_features) + 'lgbn_estimators' + str(lgbn_estimators) +
                  'CNNn_estimators' + str(CNNn_estimators) + ':\n' +
                  'lgblearning_rate' + str(lgblearning_rate) + 'CNNlearning_rate' + str(CNNlearning_rate) +
                  'CNNepochs' + str(CNNepochs) + 'batch_size' + str(batch_size) + ':\n'
                  )
        f_1.write(str(round(PRE, 4)) + '±' + str(round(PRE_err, 4)) + '\t')
        f_1.write(str(round(REC, 4)) + '±' + str(round(REC_err, 4)) + '\t')
        f_1.write(str(round(ACC, 4)) + '±' + str(round(ACC_err, 4)) + '\t')
        f_1.write(str(round(F1, 4)) + '±' + str(round(F1_err, 4)) + '\t')
        f_1.write(str(round(AUC, 4)) + '±' + str(round(AUC_err, 4)) + '\t')
        f_1.write(str(round(AUPR, 4)) + '±' + str(round(AUPR_err, 4)) + '\t\n\n')
        f_1.close()

        with open(f'./outputs/data{mm}_results.csv', mode='a+') as f:
            f_writer = csv.writer(
                f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f_writer.writerow([str(now)])
            f_writer.writerow([str(now1)])
            f_writer.writerow(["cv" + str(cv)])
            f_writer.writerow([str(parameter)])
            f_writer.writerow(["PRE:" + str(round(PRE, 4)) + "±" + str(round(PRE_err, 4))])
            f_writer.writerow(["REC:" + str(round(REC, 4)) + "±" + str(round(REC_err, 4))])
            f_writer.writerow(["ACC:" + str(round(ACC, 4)) + "±" + str(round(ACC_err, 4))])
            f_writer.writerow(["F1:" + str(round(F1, 4)) + "±" + str(round(F1_err, 4))])
            f_writer.writerow(["AUC:" + str(round(AUC, 4)) + "±" + str(round(AUC_err, 4))])
            f_writer.writerow(["AUPR:" + str(round(AUPR, 4)) + "±" + str(round(AUPR_err, 4))])


            f_writer.writerow([str("  ")])


        print('\nwrite\n')

