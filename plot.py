import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10

from analysis_result import Result

if __name__ == '__main__':

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

    for i, result_file in enumerate(['exp_384_512_16_1024-1024-1024_False_False_84_62/result', 'exp_384_512_16_1024-1024-1024_False_True_86_26/result', 'exp_384_512_16_1024-1024-1024_True_True_92_21/result']):
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
    pass


