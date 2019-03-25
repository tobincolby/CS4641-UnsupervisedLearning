# source: https://github.com/joshuamorton/Machine-Learning
# modified heavily by pshah316

import numpy as np
import argparse
from matplotlib import pyplot as plt


from sklearn.decomposition.pca import PCA as PCA
from sklearn.decomposition import FastICA as FICA
from sklearn.random_projection import GaussianRandomProjection as RandomProjection
from sklearn.feature_selection import SelectKBest as SKBest
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans as KM
from sklearn.mixture import GaussianMixture as EM
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing

from sklearn import metrics

from scipy.stats import kurtosis

def load(filename):
    with open(filename) as data:
        instances = [line for line in data if "?" not in line]

    return np.loadtxt(instances,
                      delimiter=',')

def create_dataset(name, test, train):
    training_set = load("data/" + train)
    testing_set = load("data/" + test)
    train_x, train_y = np.hsplit(training_set, [training_set[0].size-1])
    test_x, test_y = np.hsplit(testing_set, [testing_set[0].size-1])

    return train_x, train_y, test_x, test_y

def pca(name, train_x, train_y, test_x, test_y):
    compressor = PCA(n_components = train_x[1].size)
    compressor.fit(X=train_x)
    dim_train_x = compressor.transform(train_x)
    dim_test_x = compressor.transform(test_x)
    recon_err = np.zeros(train_x[1].size)
    for i in range(2,train_x[1].size):
        ccompressor = PCA(n_components=i)
        ccompressor.fit(X=train_x)
        cdim_train_x = ccompressor.transform(train_x)
        cre_train_x = ccompressor.inverse_transform(cdim_train_x)
        recon_err[i] = metrics.mean_squared_error(train_x,cre_train_x)
    print(recon_err)
    plt.plot(recon_err)
    title = plt.title("PCA Reconstruction Error")
    plt.xlabel("# Components")
    plt.ylabel("SSE")
    #plt.xlim(range(2,train_x[1].size))
    #plt.show()
    plt.savefig(name + "_out/" + title.get_text() +".png", dpi=500)
    plt.clf()

    plt.plot(compressor.explained_variance_)
    title = plt.title("PCA Eigenvalues")
    plt.xlabel("Attribute # (Sorted)")
    plt.ylabel("Eigenvalue")
    plt.xticks(range(compressor.explained_variance_.size))
    #plt.show()
    plt.savefig(name + "_out/" + title.get_text() +".png", dpi=500)
    plt.clf()

    em(name, dim_train_x, train_y, dim_test_x, test_y, add="with PCA ", max_cluster=30)
    km(name, dim_train_x, train_y, dim_test_x, test_y, add="with PCA ", max_cluster=30)
    nn(name, dim_train_x, train_y, dim_test_x, test_y, add="with PCA ")

def ica(name, train_x, train_y, test_x, test_y):
    compressor = FICA(n_components = train_x[1].size)
    compressor.fit(X=train_x)
    dim_train_x = compressor.transform(train_x)
    dim_test_x = compressor.transform(test_x)

    for i in range(train_x[1].size):
        print(kurtosis(train_x.T[i]))

    em(name, dim_train_x, train_y, dim_test_x, test_y, add="with ICA ", max_cluster=30)
    km(name, dim_train_x, train_y, dim_test_x, test_y, add="with ICA ", max_cluster=30)
    nn(name, dim_train_x, train_y, dim_test_x, test_y, add="with ICA ")

def randproj(name, train_x, train_y, test_x, test_y):
    compressor = RandomProjection(n_components=5)
    compressor.fit(X=train_x)
    dim_train_x = compressor.transform(train_x)
    dim_test_x = compressor.transform(test_x)

    recon_err = np.zeros(train_x[1].size)
    for i in range(2,train_x[1].size):
        for j in range(10):
            ccompressor = RandomProjection(n_components=i)
            ccompressor.fit(X=train_x)
            cdim_train_x = ccompressor.transform(train_x)
            cre_train_x = cdim_train_x.dot(ccompressor.components_)
            recon_err[i] = recon_err[i] + metrics.mean_squared_error(train_x,cre_train_x)
    recon_err = [i / 10 for i in recon_err]
    plt.plot(recon_err)
    title = plt.title("RP Reconstruction Error")
    plt.xlabel("# Components")
    print(train_x[1].size)
    #plt.xlim(range(2,train_x[1].size))
    plt.ylabel("SSE")
    #plt.show()
    plt.savefig(name + "_out/" + title.get_text() +".png", dpi=500)
    plt.clf()

    em(name, dim_train_x, train_y, dim_test_x, test_y, add="with RP ", max_cluster=30)
    km(name, dim_train_x, train_y, dim_test_x, test_y, add="with RP ", max_cluster=30)
    nn(name, dim_train_x, train_y, dim_test_x, test_y, add="with RP ")


def kbest(name, train_x, train_y, test_x, test_y):
    compressor = SKBest(score_func=mutual_info_classif,k=5)
    compressor.fit(X=train_x, y=train_y)
    dim_train_x = compressor.transform(train_x)
    dim_test_x = compressor.transform(test_x)

    em(name, dim_train_x, train_y, dim_test_x, test_y, add="with KB ", max_cluster=30)
    km(name, dim_train_x, train_y, dim_test_x, test_y, add="with KB ", max_cluster=30)
    nn(name, dim_train_x, train_y, dim_test_x, test_y, add="with KB ")

def em(name, train_x, train_y, test_x, test_y, add="", max_cluster=5):
    clf_loglikely_err = np.zeros(max_cluster + 1)
    clf_silhouette_err = np.zeros(max_cluster + 1)
    train_homo_err = np.zeros(max_cluster + 1)
    test_homo_err = np.zeros(max_cluster + 1)


    for i in range(2, max_cluster + 1):
        clf = EM(n_components=i)
        clf.fit(train_x)

        train_y_clf = clf.predict(train_x)
        test_y_clf = clf.predict(test_x)

        train_y.shape = (train_y.shape[0],)
        test_y.shape = (test_y.shape[0],)

        clf_loglikely_err[i] = clf.lower_bound_
        clf_silhouette_err[i] = metrics.silhouette_score(train_x, train_y_clf)
        train_homo_err[i] = metrics.homogeneity_score(train_y, train_y_clf)
        test_homo_err[i] = metrics.homogeneity_score(test_y, test_y_clf)


    fig, ax1 = plt.subplots()
    l1,=ax1.plot(clf_silhouette_err)
    l2,=ax1.plot(train_homo_err)
    l3,=ax1.plot(test_homo_err)
    plt.xlabel("Number of Components")
    ax1.set_ylabel("Metric Value")
    ax2 = ax1.twinx()
    l4,=ax2.plot(clf_loglikely_err, 'b-')
    ax2.set_ylabel("Log-likely Value")
    title = plt.title("Expected Maximization " + add)
    plt.xlim(2,max_cluster)
    plt.legend([l1,l2,l3,l4],['Silhouette Score', 'Homogeneity Score (training)', 'Homogeneity Score (testing)','Log-Likely'])
    fig.tight_layout()
    #plt.show()
    plt.savefig(name + "_out/" + title.get_text() +".png", dpi=500)
    plt.clf()
    if name == 'tic':
        clf = EM(n_components=15)
    else:
        clf = EM(n_components=12)

    clf.fit(train_x)
    train_y_clf = clf.predict(train_x)
    test_y_clf = clf.predict(test_x)

    dim_train_x = np.c_[ train_x, train_y_clf ]
    dim_test_x = np.c_[ test_x, test_y_clf ]
    nn(name, dim_train_x, train_y, dim_test_x, test_y, add="on EM" + add)

def km(name, train_x, train_y, test_x, test_y, add="", max_cluster=5):
    clf_inertia_err = np.zeros(max_cluster + 1)
    clf_silhouette_err = np.zeros(max_cluster + 1)
    train_homo_err = np.zeros(max_cluster + 1)
    test_homo_err = np.zeros(max_cluster + 1)

    for i in range(2, max_cluster + 1):
        clf = KM(n_clusters=i,max_iter=5000)
        clf.fit(train_x)

        train_y_clf = clf.predict(train_x)
        test_y_clf = clf.predict(test_x)

        train_y.shape = (train_y.shape[0],)
        test_y.shape = (test_y.shape[0],)

        clf_inertia_err[i] = clf.inertia_
        clf_silhouette_err[i] = metrics.silhouette_score(train_x, train_y_clf)
        train_homo_err[i] = metrics.homogeneity_score(train_y, train_y_clf)
        test_homo_err[i] = metrics.homogeneity_score(test_y, test_y_clf)

    fig, ax1 = plt.subplots()
    l1,=ax1.plot(clf_silhouette_err)
    l2,=ax1.plot(train_homo_err)
    l3,=ax1.plot(test_homo_err)
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Metric Value")
    ax2 = ax1.twinx()
    l4,=ax2.plot(clf_inertia_err, 'b-')
    ax2.set_ylabel("Inertia Value")
    title = plt.title("k-Means " + add)
    plt.xlim(2,max_cluster)
    plt.legend([l1,l2,l3,l4],['Silhouette Score', 'Homogeneity Score (training)', 'Homogeneity Score (testing)','Inertia'])
    fig.tight_layout()
    #plt.show()
    plt.savefig(name + "_out/" + title.get_text() +".png", dpi=500)
    plt.clf()

    if name == 'tic':
        clf = KM(n_clusters=15, max_iter=5000)
    else:
        clf = KM(n_clusters=11, max_iter=5000)
    clf.fit(train_x)
    train_y_clf = clf.predict(train_x)
    test_y_clf = clf.predict(test_x)

    dim_train_x = np.c_[ train_x, train_y_clf ]
    dim_test_x = np.c_[ test_x, test_y_clf ]
    nn(name, dim_train_x, train_y, dim_test_x, test_y, add="on KM" + add)

def nn(name, train_x, train_y, test_x, test_y, add=""):
    clf_train_err = np.zeros(10)
    clf_test_err = np.zeros(10)

    train_y.shape = (train_y.shape[0],)
    test_y.shape = (test_y.shape[0],)

    for i in range(0,10):
        clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(165,), max_iter=i*75+100)
        if name == 'car_quality':
            clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(35,), max_iter=i*75+100)

        clf.fit(X=train_x, y=train_y)
        clf_train_err[i] = 1 - clf.score(train_x, train_y)
        clf_test_err[i] = 1 - clf.score(test_x, test_y)

    plt.plot(clf_train_err)
    plt.plot(clf_test_err)
    plt.xticks(range(0,10), [100+i*75 for i in range(0,10)])
    title = plt.title("Neural Network " + add)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Error")
    plt.legend(['Training Error', 'Testing Error'])
    #plt.show()
    plt.savefig(name + "_out/" + title.get_text() +".png", dpi=500)
    plt.clf()

def vis(name, train_x, train_y, test_x, test_y, add=""):
    reduced_data = PCA(n_components=2).fit_transform(train_x)
    kmeans = KM(init='k-means++', n_clusters=6)
    h = 0.2  # point in the mesh [x_min, x_max]x[y_min, y_max].
    if name == 'tic':
        kmeans = KM(init='k-means++', n_clusters=15)
        em = EM(n_components=15)
        h = 1
    else:
        kmeans = KM(init='k-means++', n_clusters=11)
        em = EM(n_components=12)
    for k in range(0,2):
        if k==0:
            if name == 'tic':
                kmeans = KM(init='k-means++', n_clusters=15)
            else:
                kmeans = KM(init='k-means++', n_clusters=11)
        if k==1:
            if name == 'tic':
                kmeans = EM(n_components=15)
            else:
                kmeans = EM(n_components=12)
        for i in range(0,4):
            if i ==0:
                reduced_data=PCA(n_components=2).fit_transform(train_x)
            if i ==1:
                reduced_data = FICA(n_components=2).fit_transform(train_x)
            if i ==2:
                reduced_data = RandomProjection(n_components =2).fit_transform(train_x)
            if i ==3:
                reduced_data = SKBest(score_func=mutual_info_classif,k=2).fit_transform(train_x, train_y.ravel())
            kmeans.fit(reduced_data)
            print(i)
            print("made data")

            # Plot the decision boundary. For that, we will assign a color to each
            x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
            y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            print("mesh gird")
            # Obtain labels for each point in mesh. Use last trained model.
            Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
            print("labels")
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure(1)
            plt.clf()
            plt.imshow(Z, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       cmap=plt.cm.Paired,
                       aspect='auto', origin='lower')

            plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
            # Plot the centroids as a white X
            if k != 1:
                centroids = kmeans.cluster_centers_
                plt.scatter(centroids[:, 0], centroids[:, 1],
                            marker='x', s=169, linewidths=3,
                            color='w', zorder=10)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            #plt.show()
            if k == 0:
                plt.savefig(name + "_out/KM " + str(i) +".png", dpi=500)
            else:
                plt.savefig(name + "_out/EM " + str(i) +".png", dpi=500)
            plt.clf()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run clustering algorithms on stuff')
    parser.add_argument("name")
    args = parser.parse_args()
    name = args.name
    train = name+"_train.csv"
    test = name+"_test.csv"
    train_x, train_y, test_x, test_y = create_dataset(name, test, train)
    # nn(name, train_x, train_y, test_x, test_y); print('nn done')
    # em(name, train_x, train_y, test_x, test_y, max_cluster = 30); print('em done')
    # km(name, train_x, train_y, test_x, test_y, max_cluster = 30); print('km done')
    # pca(name, train_x, train_y, test_x, test_y); print('pca done')
    # ica(name, train_x, train_y, test_x, test_y); print('ica done')
    # randproj(name, train_x, train_y, test_x, test_y); print('randproj done')
    # kbest(name, train_x, train_y, test_x, test_y); print('kbest done')
    vis(name,train_x[:2000],train_y[:2000],test_x,test_y); print('vis done')
