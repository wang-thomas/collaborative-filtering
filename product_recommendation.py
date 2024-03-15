from collections import defaultdict
import numpy as np

class ProductRecommender:
    def __init__(self, input_file):
        if not input_file:
            raise ValueError("input_file is required")
        self.order_map, self.unique_skus = self._read_order_history(input_file)
        self.probability_matrix = self._create_probability_matrix(
            self.order_map, self.unique_skus
        )

    # order_map: a dictionary where the key is the transaction id and the value is a set of skus
    # unique_skus: a list of unique skus, sorted
    def _read_order_history(self, input_file):
        unique_skus = set()
        order_map = dict()
        with open(input_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                [tranid, sku] = line.split(",")
                unique_skus.add(sku)
                if tranid in order_map:
                    order_map[tranid].add(sku)
                else:
                    order_map[tranid] = {sku}
        unique_skus = list(unique_skus)
        unique_skus.sort()
        return order_map, unique_skus

    def _create_probability_matrix(self, order_map, unique_skus):
        mat = np.zeros((len(unique_skus), len(unique_skus)))
        count_map = defaultdict(int)
        for order in order_map:
            skus = list(order_map[order])
            skus.sort()
            for i, sku1 in enumerate(skus):
                count_map[(sku1, sku1)] += 1
                for j in range(i + 1, len(skus)):
                    count_map[(sku1, skus[j])] += 1

        for i, sku1 in enumerate(unique_skus):
            for j, sku2 in enumerate(unique_skus):
                joint = (sku1, sku2) if (sku1, sku2) in count_map else (sku2, sku1)
                mat[i, j] = float(count_map[joint] / count_map[(sku2, sku2)])

        return mat

    def predict(
        self,
        sku_list,
        acc_lic_only=False,
        num_recommendations=3,
    ):
        cumulative_prob = np.zeros(len(self.unique_skus))
        # consider each potential sku to recommend
        for i in range(len(self.unique_skus)):
            for sku in sku_list:
                index = self.unique_skus.index(sku)
                cumulative_prob[i] += self.probability_matrix[i, index]

        top_sku_indices = np.argsort(cumulative_prob)[::-1]
        top_sku = [self.unique_skus[i] for i in top_sku_indices][len(sku_list) :]

        if acc_lic_only:
            top_sku = [sku for sku in top_sku if "LIC" in sku or "ACC" in sku]

        return top_sku[:num_recommendations]


if __name__ == "__main__":
    input_file = "ns_query_two_year_both.csv"
    cart = ["LIC-1Y", "LIC-3Y"]
    acc_lic_only = True
    num_recommendations = 5

    product_recommender = ProductRecommender(input_file)
    predictions = product_recommender.predict(
        cart, acc_lic_only=acc_lic_only, num_recommendations=num_recommendations
    )

    print(predictions)
