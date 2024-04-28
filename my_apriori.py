import os
import shutil
import sys

from pyspark import SparkContext

# 生成候选项集
def generate_candidates(frequent_items, k):
    return set([tuple(sorted(set(a) | set(b))) for a in frequent_items for b in frequent_items if len(set(a) | set(b)) == k])


def calculate_support(sc, rdd, candidates):
    candidate_rdd = sc.broadcast(candidates)
    return rdd.flatMap(lambda x: [(c, 1) for c in candidate_rdd.value if set(c).issubset(x)]).reduceByKey(
        lambda x, y: x + y)

def apriori(sc, f_input, f_output, min_sup):
    # read the raw data
    # rdd = sc.textFile(f_input)
    rdd = sc.textFile(f_input).map(lambda x: set(x.split())).filter(lambda x: x)  # 过滤掉空行
    # count the total number of samples
    n_samples = rdd.count()
    # min_sup to frequency
    sup = n_samples * min_sup

    # Initial frequent single items
    single_items = rdd.flatMap(lambda x: [(i, 1) for i in x]).reduceByKey(lambda x, y: x + y)
    frequent_items = single_items.filter(lambda x: x[1] >= sup).map(lambda x: x[0]).collect()
    all_frequent_items = [(tuple([item]), single_items.filter(lambda x: x[0] == item).collect()[0][1]) for item in
                          frequent_items]

    k = 2
    while frequent_items:
        candidates = generate_candidates(frequent_items, k)
        if not candidates:
            break
        candidate_supports = calculate_support(sc, rdd, candidates)
        frequent_items = candidate_supports.filter(lambda x: x[1] >= sup).collect()
        if frequent_items:
            all_frequent_items.extend(frequent_items)
            frequent_items = [x[0] for x in frequent_items]
        k += 1

    # output the result to file system
    sc.parallelize(all_frequent_items, numSlices=1).saveAsTextFile(f_output)
    sc.stop()


if __name__ == "__main__":
    if os.path.exists(sys.argv[2]):
        shutil.rmtree(sys.argv[2])
    apriori(SparkContext(appName="Spark Apriori"), sys.argv[1], sys.argv[2], float(sys.argv[3]))