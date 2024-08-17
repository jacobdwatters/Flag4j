package org.flag4j.sparse_matrix;

import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooMatrix;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooMatrixSumRowColTests {

    @Test
    void sumRowsTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        VectorOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.40631, 0.37581, 0.60885};
        aRowIndices = new int[]{0, 2, 2};
        aColIndices = new int[]{4, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 5);
        expEntries = new double[]{0.37581, 1.01516};
        expRowIndices = new int[]{0, 0};
        expColIndices = new int[]{2, 4};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumRows());

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.99559, 0.37059, 0.28233, 0.82081, 0.70652};
        aRowIndices = new int[]{1, 6, 6, 8, 10};
        aColIndices = new int[]{6, 7, 12, 12, 6};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 23);
        expEntries = new double[]{1.70211, 0.37059, 1.10314};
        expRowIndices = new int[]{0, 0, 0};
        expColIndices = new int[]{6, 7, 12};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumRows());

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new double[]{0.4785, 0.46607, 0.4739, 0.40845, 0.1922, 0.13658, 0.86071, 0.03523, 0.52538, 0.54012, 0.37653, 0.54509, 0.73038, 0.8632, 0.54141, 0.63293, 0.85915, 0.84288, 0.94662, 0.8769, 0.62606, 0.61864, 0.38363, 0.22669, 0.4159, 0.01247, 0.05181, 0.19392, 0.73593, 0.38205, 0.72307, 0.46481, 0.42013, 0.71848, 0.02115, 0.7439, 0.06188, 0.50431, 0.14248, 0.70111, 0.71148, 0.43561, 0.01496, 0.43713, 0.17841, 0.76178, 0.03912, 0.81465, 0.08378, 0.13609, 0.5781, 0.62992, 0.23704, 0.90436, 0.19221, 0.98819, 0.8235, 0.5845, 0.33021, 0.72445, 0.46027, 0.01189, 0.05981, 0.43234, 0.73111, 0.22429, 0.31349, 0.30878, 0.93422, 0.61657, 0.73937, 0.93716, 0.54622, 0.50434, 0.15313, 0.47365, 0.05947, 0.30073, 0.6649, 0.31174, 0.45423, 0.91421, 0.01256, 0.94445, 0.41648, 0.0089, 0.17159, 0.24502, 0.38655, 0.6692, 0.06662, 0.4283, 0.14504, 0.0097, 0.98692, 0.55676, 0.55608, 0.31086};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{8, 16, 143, 230, 320, 403, 453, 464, 522, 534, 551, 651, 658, 680, 705, 707, 864, 976, 983, 40, 51, 89, 159, 203, 248, 254, 335, 352, 460, 587, 596, 612, 648, 652, 665, 688, 704, 705, 726, 728, 729, 770, 780, 796, 848, 870, 963, 973, 983, 54, 66, 97, 105, 145, 180, 187, 193, 251, 315, 364, 382, 389, 437, 524, 593, 678, 779, 804, 852, 962, 78, 94, 109, 124, 183, 209, 245, 388, 445, 459, 559, 631, 744, 880, 959, 974, 14, 27, 118, 151, 272, 334, 395, 402, 436, 464, 700, 809};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 1000);
        expEntries = new double[]{0.4785, 0.17159, 0.46607, 0.24502, 0.8769, 0.62606, 0.13609, 0.5781, 0.73937, 0.61864, 0.93716, 0.62992, 0.23704, 0.54622, 0.38655, 0.50434, 0.4739, 0.90436, 0.6692, 0.38363, 0.19221, 0.15313, 0.98819, 0.8235, 0.22669, 0.47365, 0.40845, 0.05947, 0.4159, 0.5845, 0.01247, 0.06662, 0.33021, 0.1922, 0.4283, 0.05181, 0.19392, 0.72445, 0.46027, 0.30073, 0.01189, 0.14504, 0.0097, 0.13658, 0.98692, 0.05981, 0.6649, 0.86071, 0.31174, 0.73593, 0.59199, 0.52538, 0.43234, 0.54012, 0.37653, 0.45423, 0.38205, 0.73111, 0.72307, 0.46481, 0.91421, 0.42013, 0.54509, 0.71848, 0.73038, 0.02115, 0.22429, 0.8632, 0.7439, 0.55608, 0.06188, 1.04572, 0.63293, 0.14248, 0.70111, 0.71148, 0.01256, 0.43561, 0.31349, 0.01496, 0.43713, 0.30878, 0.31086, 0.17841, 0.93422, 0.85915, 0.76178, 0.94445, 0.41648, 0.61657, 0.03912, 0.81465, 0.0089, 0.84288, 1.0304};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expColIndices = new int[]{8, 14, 16, 27, 40, 51, 54, 66, 78, 89, 94, 97, 105, 109, 118, 124, 143, 145, 151, 159, 180, 183, 187, 193, 203, 209, 230, 245, 248, 251, 254, 272, 315, 320, 334, 335, 352, 364, 382, 388, 389, 395, 402, 403, 436, 437, 445, 453, 459, 460, 464, 522, 524, 534, 551, 559, 587, 593, 596, 612, 631, 648, 651, 652, 658, 665, 678, 680, 688, 700, 704, 705, 707, 726, 728, 729, 744, 770, 779, 780, 796, 804, 809, 848, 852, 864, 870, 880, 959, 962, 963, 973, 974, 976, 983};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumRows());

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.63564, 0.17562, 0.86217, 0.73278};
        aRowIndices = new int[]{1, 2, 3, 4};
        aColIndices = new int[]{1, 1, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 3);
        expEntries = new double[]{0.86217, 1.5440399999999999};
        expRowIndices = new int[]{0, 0};
        expColIndices = new int[]{0, 1};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumRows());

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.26569, 0.86318, 0.39663, 0.11514};
        aRowIndices = new int[]{1, 1, 2, 3};
        aColIndices = new int[]{0, 2, 2, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 3);
        expEntries = new double[]{0.26569, 0.11514, 1.2598099999999999};
        expRowIndices = new int[]{0, 0, 0};
        expColIndices = new int[]{0, 1, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumRows());

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.04502, 0.13635, 0.06036, 0.83215};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 3);
        expEntries = new double[]{0.18137, 0.8925099999999999};
        expRowIndices = new int[]{0, 0};
        expColIndices = new int[]{1, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumRows());

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.67254, 0.44883, 0.88388, 0.91435};
        aRowIndices = new int[]{0, 1, 2, 4};
        aColIndices = new int[]{0, 1, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 3);
        expEntries = new double[]{0.67254, 0.44883, 1.79823};
        expRowIndices = new int[]{0, 0, 0};
        expColIndices = new int[]{0, 1, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumRows());
    }


    @Test
    void sumColsTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        VectorOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.75389, 0.51306, 0.48131};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{4, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 1);
        expEntries = new double[]{0.75389, 0.99437};
        expRowIndices = new int[]{0, 1};
        expColIndices = new int[]{0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumCols());

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.41181, 0.59753, 0.25686, 0.63607, 0.42909};
        aRowIndices = new int[]{1, 4, 4, 8, 10};
        aColIndices = new int[]{3, 19, 21, 14, 7};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(11, 1);
        expEntries = new double[]{0.41181, 0.85439, 0.63607, 0.42909};
        expRowIndices = new int[]{1, 4, 8, 10};
        expColIndices = new int[]{0, 0, 0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumCols());

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new double[]{0.96541, 0.23182, 0.07845, 0.96794, 0.84639, 0.53856, 0.4269, 0.78005, 0.66003, 0.70792, 0.69554, 0.37556, 0.65111, 0.78569, 0.98715, 0.21241, 0.65537, 0.82748, 0.50424, 0.99792, 0.58517, 0.24915, 0.14337, 0.51556, 0.14426, 0.32797, 0.08537, 0.0463, 0.35508, 0.94644, 0.78209, 0.81805, 0.35217, 0.27904, 0.19581, 0.17742, 0.40363, 0.71917, 0.8162, 0.8992, 0.63198, 0.45534, 0.51728, 0.10742, 0.62027, 0.32517, 0.13459, 0.08623, 0.03493, 0.99246, 0.82285, 0.90454, 0.90695, 0.26284, 0.68773, 0.66149, 0.77315, 0.62172, 0.99216, 0.92722, 0.84606, 0.36472, 0.7306, 0.72121, 0.87073, 0.58505, 0.65518, 0.90032, 0.4567, 0.79366, 0.39873, 0.11418, 0.11208, 0.661, 0.24619, 0.91351, 0.66708, 0.88694, 0.60777, 0.51422, 0.4934, 0.49826, 0.93162, 0.92324, 0.24411, 0.94333, 0.97916, 0.85603, 0.93418, 0.1998, 0.8346, 0.0968, 0.66598, 0.19976, 0.45124, 0.10531, 0.68872, 0.63419};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{76, 96, 135, 136, 287, 327, 331, 356, 375, 497, 596, 652, 770, 777, 785, 990, 33, 66, 81, 125, 219, 224, 316, 327, 389, 402, 599, 644, 648, 687, 700, 715, 731, 755, 791, 812, 818, 885, 921, 974, 3, 19, 77, 118, 205, 241, 250, 342, 388, 460, 461, 464, 484, 497, 618, 666, 717, 746, 747, 812, 861, 933, 7, 83, 157, 222, 280, 318, 359, 389, 460, 547, 559, 592, 627, 646, 738, 797, 800, 817, 857, 875, 926, 959, 103, 165, 194, 265, 267, 417, 420, 486, 558, 622, 682, 688, 811, 840};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5, 1);
        expEntries = new double[]{9.91093, 11.826459999999999, 12.6771, 13.68167, 7.833210000000001};
        expRowIndices = new int[]{0, 1, 2, 3, 4};
        expColIndices = new int[]{0, 0, 0, 0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumCols());

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.81821, 0.30375, 0.63648, 0.05163};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 2, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5, 1);
        expEntries = new double[]{1.12196, 0.63648, 0.05163};
        expRowIndices = new int[]{0, 1, 2};
        expColIndices = new int[]{0, 0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumCols());

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.57081, 0.2302, 0.24213, 0.95985};
        aRowIndices = new int[]{0, 1, 4, 4};
        aColIndices = new int[]{1, 1, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5, 1);
        expEntries = new double[]{0.57081, 0.2302, 1.20198};
        expRowIndices = new int[]{0, 1, 4};
        expColIndices = new int[]{0, 0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumCols());

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.6901, 0.22095, 0.98998, 0.56584};
        aRowIndices = new int[]{0, 0, 4, 4};
        aColIndices = new int[]{1, 2, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5, 1);
        expEntries = new double[]{0.91105, 1.55582};
        expRowIndices = new int[]{0, 4};
        expColIndices = new int[]{0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumCols());

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.71268, 0.56376, 0.88597, 0.61015};
        aRowIndices = new int[]{1, 3, 3, 4};
        aColIndices = new int[]{0, 0, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5, 1);
        expEntries = new double[]{0.71268, 1.4497300000000002, 0.61015};
        expRowIndices = new int[]{1, 3, 4};
        expColIndices = new int[]{0, 0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).toDense().toVector();

        assertEquals(exp, a.sumCols());
    }
}
