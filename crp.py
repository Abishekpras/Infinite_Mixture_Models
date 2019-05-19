import random
import numpy as np
import scipy.stats as stats


class CRP(object):
    def __init__(self, alpha):
        self.clusters = {}
        self.cluster_assn = {}
        self.alpha = alpha
        self.data = None

        self.gs_n_iters = 75
        self.emp_bayes_interval = 5
        self.emp_bayes_start = 10

    def cluster_data(self, data):
        self.data = data
        [self._cluster_data_online(cust=cust) for cust in range(len(data))]
        self._gibbs_sampler()
        return self.clusters

    def _cluster_data_online(self, cust):

        cluster_prior = [len(self.clusters[cls_id]) / (self.alpha + cust) for cls_id in self.clusters]
        cluster_prior.append(self.alpha / (self.alpha + cust))

        likelihood_wts = self.likelihood(ix=cust)
        cluster_posterior = cluster_prior * likelihood_wts

        cluster_ids = [cls_id for cls_id in self.clusters]
        new_id = 0 if cluster_ids == [] else (1 + max(cluster_ids))
        cluster_ids = cluster_ids + [new_id]
        cluster_ix = random.choices(cluster_ids, cluster_posterior)[0]
        if (cluster_ix == new_id):
            self.clusters[cluster_ix] = []
        self.clusters[cluster_ix].append(cust)
        self.cluster_assn[cust] = cluster_ix

    def _gibbs_sampler(self):

        for i in range(self.gs_n_iters):

            new_clusters = 0
            for cust in list(range(len(self.data))):

                cluster_ix = self.cluster_assn[cust]
                self.clusters[cluster_ix].remove(cust)
                if (len(self.clusters[cluster_ix]) == 0):
                    del self.clusters[cluster_ix]

                n_clusters = len(self.clusters)
                self._cluster_data_online(cust=cust)
                new_clusters += (len(self.clusters) - n_clusters)

            if (i % self.emp_bayes_interval == 0) and (i > self.emp_bayes_start):
                self.alpha = new_clusters if new_clusters else 1

    def likelihood(self, ix):

        mean = [np.mean(self.data[self.clusters[cls_id]]) for cls_id in self.clusters]
        mean = np.append(mean, 0)
        std = [1 for cluster in self.clusters]
        std = np.append(std, 10)
        ll = stats.norm.pdf(self.data[ix], mean, std)
        return ll / sum(ll)

    def print_output(self, cluster_output, true_mean):

        pred_means = [np.mean(self.data[self.clusters[cls_id]]) for cls_id in self.clusters]
        print("True Means - {}".format(true_mean))
        print("Predicted Means - {}".format(pred_means))
        print(cluster_output)


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    for alpha in [1, 5, 10]:
        print("\nAlpha = {}".format(alpha))

        r1, r2 = stats.norm.rvs(0, 1, 5), stats.norm.rvs(3, 1, 5)
        true_mean = (np.mean(r1), np.mean(r2))
        data = np.concatenate((r1, r2))

        crp = CRP(alpha)
        crp.print_output(crp.cluster_data(data), true_mean)
