import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10

from analysis_result import Result

def experminents_plot():
    plt.rcParams.update({'font.size': 25})
    fig1, axis = plt.subplots(10, 3)
    fig1.set_figwidth(30)
    fig1.set_figheight(100)
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = 300

    # fig2, axis2 = plt.subplots()
    # fig2.set_figwidth(20)
    # fig2.set_figheight(10)
    # plt.tight_layout()
    # plt.rcParams['figure.dpi'] = 300

    fig3, axis3 = plt.subplots()
    fig3.set_figwidth(20)
    fig3.set_figheight(10)
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = 300
    for i, result_file in enumerate(['exp_384_512_16_1024-1024-1024_False_False_84_62/result',
                                     'exp_384_512_16_1024-1024-1024_False_True_86_26/result',
                                     'exp_384_512_16_1024-1024-1024_True_True_92_21/result']):
        result = Result(result_file)
        result = result.load()

        if i == 0:
            cifar10 = CIFAR10(root='', download=True)
            roc_list = []

            for _class in range(10):
                roc_list += [result.class_to_rotation_roc[_class][0]]
                emb = result.tsne_plots[_class]['emb']
                anomaly_labels = result.tsne_plots[_class]['anomaly_labels']
                rot_0_labels = result.tsne_plots[_class]['rot_0_labels']
                ax = axis[_class, i]
                ax.scatter(emb[anomaly_labels, 0], emb[anomaly_labels, 1], label='anomaly', c='r', marker='o')
                ax.scatter(emb[rot_0_labels, 0], emb[rot_0_labels, 1], label='nomaly', c='g', marker='o')
                ax.set_title(f'{cifar10.classes[_class]}')
                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
                ax.grid(alpha=0.75)
                # ax.legend()

                if _class == 9:
                    handles, labels = ax.get_legend_handles_labels()
                    fig1.legend(handles, labels, loc=(0, .99))

            np.set_printoptions(precision=1)
            print(f'roc_list: {np.array(roc_list) * 100}, avg: {np.array(roc_list).mean()}')
            # plt.tight_layout()
            # plt.show()
        else:
            # plt.rcParams.update({'font.size': 25})
            # fig, axis = plt.subplots(2, 5)
            # fig.set_figwidth(50)
            # fig.set_figheight(20)
            cifar10 = CIFAR10(root='', download=True)
            roc_list = []

            for _class in range(10):
                _l = []
                for _, values in result.class_to_rotation_roc[_class].items():
                    _l.append(values)
                roc_list += [_l]
                emb = result.tsne_plots[_class]['emb']
                anomaly_labels = result.tsne_plots[_class]['anomaly_labels']
                rot_0_labels = result.tsne_plots[_class]['rot_0_labels']
                rot_90_labels = result.tsne_plots[_class]['rot_90_labels']
                rot_180_labels = result.tsne_plots[_class]['rot_180_labels']
                rot_270_labels = result.tsne_plots[_class]['rot_270_labels']
                ax = axis[_class, i]
                ax.scatter(emb[anomaly_labels, 0], emb[anomaly_labels, 1], label='anomaly', c='r', marker='o')
                ax.scatter(emb[rot_0_labels, 0], emb[rot_0_labels, 1], label='0 rotation', c='g', marker='o')
                ax.scatter(emb[rot_90_labels, 0], emb[rot_90_labels, 1], label='90 rotation', c='k', marker='o')
                ax.scatter(emb[rot_180_labels, 0], emb[rot_180_labels, 1], label='180 rotation', c='m', marker='o')
                ax.scatter(emb[rot_270_labels, 0], emb[rot_270_labels, 1], label='270 rotation', c='y', marker='o')
                ax.set_title(f'{cifar10.classes[_class]}')
                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
                ax.grid(alpha=0.75)
                # ax.legend()

                if _class == 9:
                    handles, labels = ax.get_legend_handles_labels()
                    if i == 1:
                        fig1.legend(handles, labels, loc=(.355, .973))
                    if i == 2:
                        fig1.legend(handles, labels, loc=(.685, .973))

            roc_list = np.array(roc_list) * 100

            print(f'roc_list: {roc_list[:, -1]}, avg: {roc_list[:, -1].mean()}')
            # plt.show()
            # if i == 2:
            #     plt.show()

            # if i == 1:
            #     # plt.figure(figsize=(20, 10))
            #     axis2.plot(np.arange(1, 5), roc_list.mean(axis=0))
            #     axis2.set_xticks(np.arange(1, 5), ['{0}', '{0, 90}', '{0, 90, 180}', '{0, 90, 180, 270}'])
            #     axis2.grid(alpha=0.75)
            #     # axis2.tight_layout()
            #     axis2.set_xlabel('cumulative score over set of rotated ocsvm')
            #     axis2.set_ylabel('roc')
            #     # plt.tight_layout()
            if i == 1:
            # if i == 2:
                # plt.figure(figsize=(20, 10))
                axis3.plot(np.arange(1, 5), roc_list.mean(axis=0))
                axis3.set_xticks(np.arange(1, 5), ['{0}', '{0, 90}', '{0, 90, 180}', '{0, 90, 180, 270}'])
                axis3.grid(alpha=0.75)
                # axis2.tight_layout()
                axis3.set_xlabel('cumulative score over set of rotated ocsvm')
                axis3.set_ylabel('roc')
                # plt.tight_layout()
    plt.tight_layout()
    plt.show()

def batch_hyperparameter_plot():
    plt.rcParams.update({'font.size': 25})
    batches = [64, 128, 256, 384, 512]
    cifar10 = CIFAR10(root='.', download=True)

    if True:
        fig, axis = plt.subplots()
        fig.set_figwidth(20)
        fig.set_figheight(10)
        plt.tight_layout()

        fig2, axis2 = plt.subplots()
        fig2.set_figwidth(20)
        fig2.set_figheight(10)
        plt.tight_layout()



        plt.rcParams['figure.dpi'] = 300

        batch_roc_list = []
        for batch in batches:
            result = Result(f'/home/zia/exp_collection/exp_{batch}_512_16_1024-1024-1024/result')
            result = result.load()

            roc_list = []
            if hasattr(result, 'class_to_rotation_roc'):
                for _class in range(10):
                    _l = []
                    if result.class_to_rotation_roc.get(_class):
                        for _, values in result.class_to_rotation_roc[_class].items():
                            _l.append(values)
                    else:
                        _l += [0, 0, 0, 0]
                    roc_list += [_l]
            else:
                roc_list = [[0, 0, 0, 0]] *10


            batch_roc_list.append(np.array(roc_list)[:, -1].tolist())

        batch_roc_list = np.array(batch_roc_list)

        x_axis = np.arange(len(batches))*30
        for _class in range(10):
            axis.bar(x_axis + _class*2, batch_roc_list[:, _class], width=2, label=f'class: {cifar10.classes[_class]}')

        axis.set_xticks(x_axis+10, batches)
        axis.legend()
        axis.set_xlabel('batch_size')
        axis.set_ylabel('roc')
        axis.grid(alpha=0.75)

        axis2.plot(np.array(batches), np.mean(batch_roc_list, axis=1) * np.logical_not(np.any(batch_roc_list == 0, axis=1)), linewidth=5, marker='o', markersize=10, mfc='b', label='mean roc')
        axis2.set_xticks(batches, batches)
        axis2.legend()
        axis2.set_xlabel('batch_size')
        axis2.set_ylabel('roc')
        axis2.grid(alpha=0.75)
        plt.show()

    if False:
        plt.rcParams.update({'font.size': 25})
        fig, axis = plt.subplots(10, 5)
        fig.set_figwidth(5*9)
        fig.set_figheight(10*9)
        plt.tight_layout()
        plt.rcParams['figure.dpi'] = 300
        for i, batch in enumerate(batches):
            result = Result(f'/home/zia/exp_collection/exp_{batch}_512_16_1024-1024-1024/result')
            result = result.load()

            roc_list = []
            if hasattr(result, 'class_to_rotation_roc'):
                for _class in range(10):
                    _l = []
                    if result.class_to_rotation_roc.get(_class):
                        for _, values in result.class_to_rotation_roc[_class].items():
                            _l.append(values)
                        roc_list += [_l]
                        emb = result.tsne_plots[_class]['emb']
                        anomaly_labels = result.tsne_plots[_class]['anomaly_labels']
                        rot_0_labels = result.tsne_plots[_class]['rot_0_labels']
                        rot_90_labels = result.tsne_plots[_class]['rot_90_labels']
                        rot_180_labels = result.tsne_plots[_class]['rot_180_labels']
                        rot_270_labels = result.tsne_plots[_class]['rot_270_labels']
                        ax = axis[_class, i]
                        ax.scatter(emb[anomaly_labels, 0], emb[anomaly_labels, 1], label='anomaly', c='r', marker='o')
                        ax.scatter(emb[rot_0_labels, 0], emb[rot_0_labels, 1], label='0 rotation', c='g', marker='o')
                        ax.scatter(emb[rot_90_labels, 0], emb[rot_90_labels, 1], label='90 rotation', c='k', marker='o')
                        ax.scatter(emb[rot_180_labels, 0], emb[rot_180_labels, 1], label='180 rotation', c='m', marker='o')
                        ax.scatter(emb[rot_270_labels, 0], emb[rot_270_labels, 1], label='270 rotation', c='y', marker='o')
                        ax.set_title(f'{cifar10.classes[_class]}')
                        # ax.get_xaxis().set_visible(False)
                        # ax.get_yaxis().set_visible(False)
                        ax.grid(alpha=0.75)
                        # ax.legend()

                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc=(.4/2 * i, .973))

        plt.show()

def batch_hyperparameter_plot_2():
    plt.rcParams.update({'font.size': 25})
    encoding = [8, 16, 32, 64, 128]
    cifar10 = CIFAR10(root='.', download=True)

    if False:
        fig, axis = plt.subplots()
        fig.set_figwidth(20)
        fig.set_figheight(10)
        plt.tight_layout()

        plt.rcParams['figure.dpi'] = 300

        encoding_roc_list = []
        for encode in encoding:
            result = Result(f'/home/zia/exp_collection/exp_512_512_{encode}_1024-1024-1024/result')
            result = result.load()

            roc_list = []
            if hasattr(result, 'class_to_rotation_roc'):
                for _class in range(10):
                    _l = []
                    if result.class_to_rotation_roc.get(_class):
                        for _, values in result.class_to_rotation_roc[_class].items():
                            _l.append(values)
                    else:
                        _l += [0, 0, 0, 0]
                    roc_list += [_l]
            else:
                roc_list = [[0, 0, 0, 0]] *10


            encoding_roc_list.append(np.array(roc_list)[:, -1].tolist())

        encoding_roc_list = np.array(encoding_roc_list) * 100

        axis.plot(np.array(encoding), np.mean(encoding_roc_list, axis=1), linewidth=5, marker='o', markersize=10, mfc='b', label='mean roc')
        axis.set_xticks(encoding, encoding)
        axis.legend()
        axis.set_xlabel('encoding dimension')
        axis.set_ylabel('mean roc')
        axis.grid(alpha=0.75)

        plt.show()

    if True:
        plt.rcParams.update({'font.size': 25})
        fig, axis = plt.subplots(10, 5)
        fig.set_figwidth(5*9)
        fig.set_figheight(10*9)
        plt.tight_layout()
        plt.rcParams['figure.dpi'] = 300
        for i, encode in enumerate(encoding):
            result = Result(f'/home/zia/exp_collection/exp_512_512_{encode}_1024-1024-1024/result')
            result = result.load()

            roc_list = []
            if hasattr(result, 'class_to_rotation_roc'):
                for _class in range(10):
                    _l = []
                    if result.class_to_rotation_roc.get(_class):
                        for _, values in result.class_to_rotation_roc[_class].items():
                            _l.append(values)
                        roc_list += [_l]
                        emb = result.tsne_plots[_class]['emb']
                        anomaly_labels = result.tsne_plots[_class]['anomaly_labels']
                        rot_0_labels = result.tsne_plots[_class]['rot_0_labels']
                        rot_90_labels = result.tsne_plots[_class]['rot_90_labels']
                        rot_180_labels = result.tsne_plots[_class]['rot_180_labels']
                        rot_270_labels = result.tsne_plots[_class]['rot_270_labels']
                        ax = axis[_class, i]
                        ax.scatter(emb[anomaly_labels, 0], emb[anomaly_labels, 1], label='anomaly', c='r', marker='o')
                        ax.scatter(emb[rot_0_labels, 0], emb[rot_0_labels, 1], label='0 rotation', c='g', marker='o')
                        ax.scatter(emb[rot_90_labels, 0], emb[rot_90_labels, 1], label='90 rotation', c='k', marker='o')
                        ax.scatter(emb[rot_180_labels, 0], emb[rot_180_labels, 1], label='180 rotation', c='m', marker='o')
                        ax.scatter(emb[rot_270_labels, 0], emb[rot_270_labels, 1], label='270 rotation', c='y', marker='o')
                        ax.set_title(f'{cifar10.classes[_class]}')
                        # ax.get_xaxis().set_visible(False)
                        # ax.get_yaxis().set_visible(False)
                        ax.grid(alpha=0.75)
                        # ax.legend()

                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc=(.4/2 * i, .973))

        plt.show()



if __name__ == '__main__':

    # experminents_plot()
    # batch_hyperparameter_plot()
    batch_hyperparameter_plot_2()



    pass


