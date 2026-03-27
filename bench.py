import matplotlib.pyplot as plt

sizes = [0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000]

data = {
    "reduction_f64":            [57791113, 85716465, 139962297, 246841739, 474924879, 950022925, 1897570456, 6841253869, 13680369078],
    "reduction_align_f64":      [57663513, 84618932, 138862752, 246462790, 476281985, 938571161, 1896797813, 6841644461, 13680048522],
    "reduction_align_64n_f64":  [57479820, 84377230, 137770680, 245530487, 477221071, 937208031, 1890919215, 6842307354, 13681426239],
    "std::reduce":              [234011813, 699168637, 1986156022, 4537107917, 9677014767, 19921362104, 40428319639, 81427848498, 163484196139],
}

labels = [hex(s) for s in sizes]

for name, times in data.items():
    ms = [t / 1e6 for t in times]
    plt.plot(labels, ms, marker="o", label=name)

plt.xlabel("Size")
plt.ylabel("Time (ms)")
plt.title("Reduction Benchmark")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bench.png", dpi=150)
plt.show()
