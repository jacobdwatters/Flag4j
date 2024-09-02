package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.arrays_old.sparse.CsrMatrixOld;
import org.flag4j.arrays.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrMatrixGetRowColTests {
    CsrMatrixOld A;
    Shape aShape;
    double[] aEntries;
    int[] aRowPointers;
    int[] aColIndices;

    CooVectorOld exp;
    int expSize;
    double[] expEntries;
    int[] expIndices;

    @Test
    void getRowTests() {
        // --------------------- sub-case 1 ---------------------
        aShape = new Shape(15, 15);
        aEntries = new double[]{0.9444819432546215, 0.5152641390019522, 0.04634772495131301, 0.14803215820360016, 0.2048101606674394, 0.6558319561786778, 0.9069760814449676, 0.030477997216917907, 0.9387977919558156, 0.5238023429188244, 0.8433349596093099, 0.14798314289004177, 0.03427785391594995, 0.15758460718502976, 0.466167291255572, 0.4737316296572681, 0.3594861179722788, 0.08236800714849835, 0.6032971123446527, 0.6843997944143437, 0.031580288610778995, 0.7568084757432743, 0.29891953261564785, 0.14333804301761444, 0.680281424827022, 0.5940742809907716, 0.9626897906443216, 0.9234721547065801, 0.5540651576730087, 0.23592539991286443, 0.6887631587880733, 0.9164023784360789, 0.1743551067783895, 0.7366695951518021};
        aRowPointers = new int[]{0, 2, 2, 5, 8, 9, 16, 18, 18, 19, 21, 22, 25, 26, 30, 34};
        aColIndices = new int[]{8, 14, 1, 3, 12, 0, 1, 14, 0, 0, 1, 3, 4, 6, 7, 10, 0, 13, 12, 0, 7, 8, 0, 9, 11, 7, 7, 11, 12, 13, 3, 5, 9, 10};
        A = new CsrMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 15;
        expEntries = new double[]{0.5238023429188244, 0.8433349596093099, 0.14798314289004177, 0.03427785391594995, 0.15758460718502976, 0.466167291255572, 0.4737316296572681};
        expIndices = new int[]{0, 1, 3, 4, 6, 7, 10};
        exp = new CooVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getRow(5));

        // --------------------- sub-case 2 ---------------------
        aShape = new Shape(201, 235);
        aEntries = new double[]{0.9411, 0.83966, 0.59873, 0.02352, 0.14564, 0.71717, 0.56019, 0.41884, 0.36276, 0.2836, 0.4199, 0.05837, 0.54325, 0.82817, 0.98986, 0.15628, 0.34137, 0.19382, 0.02578, 0.09695, 0.58898, 0.80639, 0.36464, 0.71623, 0.89614, 0.09081, 0.66702, 0.9455, 0.45062, 0.06363, 0.74842, 0.54862, 0.20503, 0.34173, 0.50337, 0.7217, 0.64544, 0.66579, 0.42057, 0.30252, 0.50443, 0.82403, -9.24, 3500.1, 35.6, 1.551, 0.64861, 0.38296, 0.12566, 0.89654, 0.01136};
        aRowPointers = new int[]{0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 11, 11, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 15, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 28, 28, 28, 29, 30, 30, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 39, 39, 41, 41, 41, 42, 42, 42, 42, 47, 47, 47, 47, 47, 47, 47, 48, 48, 49, 51, 51, 51, 51, 51};
        aColIndices = new int[]{188, 98, 52, 207, 49, 190, 110, 1, 193, 160, 135, 212, 88, 24, 119, 86, 4, 127, 186, 206, 144, 99, 7, 113, 185, 66, 38, 171, 109, 104, 233, 14, 222, 210, 231, 124, 197, 50, 211, 63, 159, 19, 0, 51, 52, 53, 95, 207, 40, 178, 213};
        A = new CsrMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 235;
        expEntries = new double[]{-9.24, 3500.1, 35.6, 1.551, 0.64861};
        expIndices = new int[]{0, 51, 52, 53, 95};
        exp = new CooVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getRow(186));

        // --------------------- sub-case 3 ---------------------
        A = new CsrMatrixOld(1000, 15235);
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(-1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(1001));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(-4));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(20015));
    }


    @Test
    void getRowAfterTests() {
        // --------------------- sub-case 1 ---------------------
        aShape = new Shape(15, 15);
        aEntries = new double[]{0.9444819432546215, 0.5152641390019522, 0.04634772495131301, 0.14803215820360016, 0.2048101606674394, 0.6558319561786778, 0.9069760814449676, 0.030477997216917907, 0.9387977919558156, 0.5238023429188244, 0.8433349596093099, 0.14798314289004177, 0.03427785391594995, 0.15758460718502976, 0.466167291255572, 0.4737316296572681, 0.3594861179722788, 0.08236800714849835, 0.6032971123446527, 0.6843997944143437, 0.031580288610778995, 0.7568084757432743, 0.29891953261564785, 0.14333804301761444, 0.680281424827022, 0.5940742809907716, 0.9626897906443216, 0.9234721547065801, 0.5540651576730087, 0.23592539991286443, 0.6887631587880733, 0.9164023784360789, 0.1743551067783895, 0.7366695951518021};
        aRowPointers = new int[]{0, 2, 2, 5, 8, 9, 16, 18, 18, 19, 21, 22, 25, 26, 30, 34};
        aColIndices = new int[]{8, 14, 1, 3, 12, 0, 1, 14, 0, 0, 1, 3, 4, 6, 7, 10, 0, 13, 12, 0, 7, 8, 0, 9, 11, 7, 7, 11, 12, 13, 3, 5, 9, 10};
        A = new CsrMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 15-4;
        expEntries = new double[]{0.03427785391594995, 0.15758460718502976, 0.466167291255572, 0.4737316296572681};
        expIndices = new int[]{0, 6-4, 7-4, 10-4};
        exp = new CooVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getRowAfter(4, 5));

        // --------------------- sub-case 2 ---------------------
        aShape = new Shape(201, 235);
        aEntries = new double[]{0.9411, 0.83966, 0.59873, 0.02352, 0.14564, 0.71717, 0.56019, 0.41884, 0.36276, 0.2836, 0.4199, 0.05837, 0.54325, 0.82817, 0.98986, 0.15628, 0.34137, 0.19382, 0.02578, 0.09695, 0.58898, 0.80639, 0.36464, 0.71623, 0.89614, 0.09081, 0.66702, 0.9455, 0.45062, 0.06363, 0.74842, 0.54862, 0.20503, 0.34173, 0.50337, 0.7217, 0.64544, 0.66579, 0.42057, 0.30252, 0.50443, 0.82403, -9.24, 3500.1, 35.6, 1.551, 0.64861, 0.38296, 0.12566, 0.89654, 0.01136};
        aRowPointers = new int[]{0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 11, 11, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 15, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 28, 28, 28, 29, 30, 30, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 39, 39, 41, 41, 41, 42, 42, 42, 42, 47, 47, 47, 47, 47, 47, 47, 48, 48, 49, 51, 51, 51, 51, 51};
        aColIndices = new int[]{188, 98, 52, 207, 49, 190, 110, 1, 193, 160, 135, 212, 88, 24, 119, 86, 4, 127, 186, 206, 144, 99, 7, 113, 185, 66, 38, 171, 109, 104, 233, 14, 222, 210, 231, 124, 197, 50, 211, 63, 159, 19, 0, 51, 52, 53, 95, 207, 40, 178, 213};
        A = new CsrMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 235-52;
        expEntries = new double[]{35.6, 1.551, 0.64861};
        expIndices = new int[]{0, 53-52, 95-52};
        exp = new CooVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getRowAfter(52, 186));

        // --------------------- sub-case 3 ---------------------
        A = new CsrMatrixOld(1000, 15235);
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRowAfter(32, -1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRowAfter(552, 1001));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRowAfter(-2, -4));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRowAfter(523523, 20015));
    }


    @Test
    void getColTests() {
        // --------------------- sub-case 1 ---------------------
        aShape = new Shape(15, 15);
        aEntries = new double[]{0.25301, 0.53231, 0.07082, 0.18391, 0.99342, 0.91803, 0.55773, 0.69665, 0.75968, 0.82797, 0.80141, 0.20495, 0.45622, 0.55735, 0.2587, 0.21637, 0.66179, 0.07654, 0.78354, 0.40217, 0.07444, 0.23364, 0.37474, 0.25595, 0.86631, 0.85896, 0.90135, 0.26993, 0.0476, 0.80036, 0.07541, 0.84697, 0.38157, 0.80478, 0.03183, 0.94379, 0.25196, 0.40963, 0.03382, 0.70915, 0.42617, 0.29418, 0.02039, 0.93004, 0.43602, 0.95209, 0.25939, 0.34429, 0.68747, 0.33029, 0.38113, 0.05704, 0.02883, 0.78966, 0.85872, 0.68404};
        aRowPointers = new int[]{0, 3, 7, 10, 14, 17, 20, 25, 29, 33, 39, 48, 52, 53, 55, 56};
        aColIndices = new int[]{3, 7, 8, 3, 8, 10, 11, 0, 7, 11, 7, 10, 12, 14, 0, 12, 14, 5, 10, 12, 0, 1, 4, 7, 10, 5, 6, 9, 14, 0, 4, 7, 13, 0, 3, 5, 8, 10, 12, 0, 1, 2, 5, 6, 7, 9, 11, 13, 2, 10, 11, 13, 12, 2, 11, 2};
        A = new CsrMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 15;
        expEntries = new double[]{0.25301, 0.18391, 0.03183};
        expIndices = new int[]{0, 1, 9};
        exp = new CooVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getCol(3));

        // --------------------- sub-case 2 ---------------------
        aShape = new Shape(130, 156);
        aEntries = new double[]{0.56645, 0.3208, 0.63297, 0.18153, 0.11088, 0.09497, 0.04357, 0.54901, 0.38698, 0.39877, 0.76046, 0.24357, 0.50073, 0.12766, 0.89106, 0.82175, 0.82884, 0.5386, 0.70469, 0.42161, 0.66371, 0.69894, 0.6243, 0.52049, 0.62234, 0.32138, 0.66633, 0.55179, 0.07566, 0.51179, 0.43311, 0.54932, 0.45256, 0.52468, 0.38453, 0.05833, 0.86735, 0.11786, 0.25418, 0.68452, 0.55614, 0.90185, 0.20837, 0.04103, 0.93511, 0.82682, 0.57889, 0.6083, 0.57669, 0.9561, 0.95932, 0.55719, 0.59588, 0.81423, 0.17855, 0.21048, 0.09767, 0.00033, 0.90339, 0.84219, 0.06671, 0.55476, 0.24247, 0.62003, 0.51283, 0.71196, 0.71011, 0.39023, 0.26949, 0.77537, 0.47517, 0.19812, 0.21502, 0.39389, 0.38716, 0.05681, 0.60245, 0.85376, 0.23729, 0.8626, 0.16511, 0.38024, 0.90325, 0.12263, 0.22186, 0.92217, 0.14853, 0.31143, 0.98065, 0.59018, 0.08668, 0.75582, 0.34995, 0.18522, 0.21834, 0.90269, 0.59142, 0.08696, 0.12282, 0.82589, 0.82838, 0.53542, 0.04364, 0.59222, 0.91171, 0.46297, 0.02174, 0.76665, 0.82138, 0.49229, 0.15752, 0.53769, 0.68729, 0.12163, 0.81868, 0.2375, 0.94655, 0.32098, 0.59418, 0.35965, 0.07267, 0.36874, 0.02055, 0.43246, 0.19222, 0.74764, 0.39322, 0.13741, 0.22517, 0.5442, 0.65535, 0.47863, 0.67543, 0.30139, 0.49645, 0.69653, 0.00726, 0.52244, 0.94099, 0.83674, 0.25175, 0.70455, 0.60379, 0.77956, 0.64457, 0.281, 0.82033, 0.2328, 0.98015, 0.3195, 0.22907, 0.53684, 0.77862, 0.5623, 0.83983, 0.09953, 0.15848, 0.75101, 0.92302, 0.71481, 0.64573, 0.81549, 0.1681, 0.14835, 0.95422, 0.08164, 0.50047, 0.95698, 0.62875, 0.90845, 0.88803, 0.10386, 0.54518, 0.70376, 0.19855, 0.61868, 0.84056, 0.35439, 0.54341, 0.7381, 0.94307, 0.02222, 0.92274, 0.9691, 0.32862, 0.79515, 0.76647, 0.94415, 0.19098, 0.61428, 0.30311, 0.78811, 0.69197, 0.35088, 0.18503, 0.27884, 0.61483, 0.943, 0.61963, 0.32168, 0.93139, 0.64034, 0.91169};
        aRowPointers = new int[]{0, 0, 0, 1, 4, 5, 6, 8, 8, 10, 10, 10, 11, 13, 15, 17, 19, 22, 22, 24, 25, 26, 27, 27, 28, 31, 32, 33, 34, 35, 37, 37, 38, 40, 41, 43, 44, 46, 46, 47, 47, 50, 52, 52, 53, 55, 57, 60, 61, 62, 64, 67, 70, 72, 72, 75, 80, 82, 82, 84, 85, 88, 89, 91, 91, 93, 95, 98, 99, 99, 101, 101, 103, 105, 107, 108, 108, 110, 115, 118, 120, 121, 121, 123, 124, 126, 126, 131, 131, 132, 134, 136, 137, 138, 138, 142, 144, 145, 150, 153, 155, 157, 160, 161, 163, 163, 164, 166, 166, 170, 171, 172, 172, 173, 175, 176, 180, 181, 184, 187, 192, 193, 194, 194, 194, 194, 197, 197, 198, 201, 203};
        aColIndices = new int[]{48, 108, 116, 125, 46, 22, 18, 84, 9, 122, 112, 0, 31, 106, 147, 67, 148, 29, 76, 25, 57, 149, 46, 151, 127, 90, 140, 28, 1, 31, 95, 131, 77, 3, 19, 114, 154, 13, 133, 148, 94, 8, 125, 135, 21, 43, 131, 30, 36, 55, 60, 133, 8, 46, 74, 14, 68, 39, 41, 83, 19, 51, 16, 109, 17, 142, 146, 52, 116, 134, 6, 95, 26, 45, 57, 2, 90, 122, 126, 131, 49, 92, 54, 105, 75, 61, 99, 108, 52, 13, 144, 49, 99, 5, 36, 24, 28, 130, 154, 6, 20, 99, 102, 84, 109, 46, 126, 67, 40, 85, 10, 23, 57, 102, 148, 80, 127, 148, 55, 127, 4, 56, 141, 125, 45, 148, 42, 84, 95, 106, 131, 97, 72, 83, 37, 149, 125, 98, 26, 86, 98, 145, 100, 152, 144, 67, 79, 80, 84, 88, 31, 84, 143, 14, 42, 86, 122, 31, 111, 118, 132, 64, 120, 105, 22, 74, 37, 56, 123, 131, 124, 129, 62, 95, 147, 146, 65, 110, 125, 127, 99, 3, 9, 131, 49, 55, 136, 4, 52, 63, 95, 126, 42, 130, 21, 97, 152, 22, 7, 31, 50, 105, 123};
        A = new CsrMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 130;
        expEntries = new double[]{0.18153, 0.20837, 0.43246, 0.00726, 0.54341};
        expIndices = new int[]{3, 34, 83, 91, 115};
        exp = new CooVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getCol(125));

        // --------------------- sub-case 3 ---------------------
        A = new CsrMatrixOld(100235, 15235);
        assertThrows(IndexOutOfBoundsException.class, ()->A.getCol(-1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getCol(15235));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getCol(-4));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getCol(222356));
    }
}
