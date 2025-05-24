import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def feature_selection(input_csv, flag):
    df = pd.read_csv(input_csv)
    feature_pixels = df['feature'].apply(lambda x: list(map(np.float32, x.split()))).to_list()
    label = df['label'].values
    file = df['file'].values
    feature_pixels = np.array(feature_pixels)
    print('feature_pixel:', feature_pixels.shape)
    feature_PCA = PCA_selection(feature_pixels, components=200)
    PCA_str = [' '.join(map(str, row)) for row in feature_PCA]
    feature_LDA = LDA_selection(feature_pixels, label, flag)
    LDA_str = [' '.join(map(str, row)) for row in feature_LDA]
    
    filename = os.path.splitext(os.path.basename(input_csv))[0]
    
    df_pca = pd.DataFrame({'file':file, 'selected': PCA_str, 'label': label})
    df_pca.to_csv(filename + '_pca.csv', index=False)
    
    df_lda = pd.DataFrame({'file':file, 'selected': LDA_str, 'label': label})
    df_lda.to_csv(filename + '_lda.csv', index=False)
    # tsne_visualize(feature_PCA, feature_LDA, label)
    
def PCA_selection(feature_pixels, components):
    pca = PCA(n_components=components)
    feature_PCA = pca.fit_transform(feature_pixels)
    print('pca:', feature_PCA.shape)
    return feature_PCA
    
def LDA_selection(feature_pixels, label, flag):
    if (flag == 'fer2013' or flag == 'raf-db'):
        components = 6
    elif(flag == 'fer-plus'):
        components = 7
    lda = LDA(n_components=components)
    feature_LDA = lda.fit_transform(feature_pixels, label)
    print('lda', feature_LDA.shape)
    return feature_LDA
def tsne_visualize(feature_PCA, feature_LDA, y):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    tsne1 = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    
    pca_tsne = tsne1.fit_transform(feature_PCA)
    lda_tsne = tsne2.fit_transform(feature_LDA)
    scatter1 = axs[0].scatter(pca_tsne[:, 0], pca_tsne[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    axs[0].set_title('PCA visualization')
    axs[0].set_xlabel('PCA t-SNE component 1')
    axs[0].set_ylabel('PCA t-SNE component 2')
    
    scatter2 = axs[1].scatter(lda_tsne[:, 0], lda_tsne[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    axs[1].set_title('LDA visualization')
    axs[1].set_xlabel('LDA t-SNE component 1')
    axs[1].set_ylabel('LDAt-SNE component 2')
    
    fig.legend(*scatter1.legend_elements(), title='Classes', loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.tight_layout()
    plt.show()
    
feature_selection('feature_csv\\fer_train_hog.csv', 'fer-plus')
# feature_selection('feature_csv\\fer_test_hog.csv', 'fer-plus')
# feature_selection('feature_csv\\rafdb_train_hog.csv', 'raf-db')
# feature_selection('feature_csv\\rafdb_test_hog.csv', 'raf-db')