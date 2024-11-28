package org.flag4j.sparse_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooMatrixToStringTests {


    @Test
    void setSliceTest() {
        String exp;

        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.8161, 0.77635, 0.73286};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{0, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);
        exp = "shape: (5, 3)\n" +
                "Non-zero data: [ 0.8161  0.77635  0.73286 ]\n" +
                "Row Indices: [0, 0, 1]\n" +
                "Col Indices: [0, 1, 2]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.5243, 0.28762, 0.17566, 0.32968, 0.44542};
        aRowIndices = new int[]{2, 4, 4, 7, 8};
        aColIndices = new int[]{13, 4, 20, 15, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);
        exp = "shape: (11, 23)\n" +
                "Non-zero data: [ 0.5243  0.28762  0.17566  0.32968  0.44542 ]\n" +
                "Row Indices: [2, 4, 4, 7, 8]\n" +
                "Col Indices: [13, 4, 20, 15, 0]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new double[]{0.06813, 0.43027, 0.27489, 0.94196, 0.30043, 0.4879, 0.99068, 0.50667, 0.91951};
        aRowIndices = new int[]{0, 0, 1, 1, 3, 3, 4, 4, 4};
        aColIndices = new int[]{557, 624, 336, 747, 125, 344, 113, 306, 350};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);
        exp = "shape: (5, 1000)\n" +
                "Non-zero data: [ 0.06813  0.43027  0.27489  0.94196  0.30043  0.4879  0.99068  0.50667  0.91951 ]\n" +
                "Row Indices: [0, 0, 1, 1, 3, 3, 4, 4, 4]\n" +
                "Col Indices: [557, 624, 336, 747, 125, 344, 113, 306, 350]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.23804, 0.38857, 0.94397, 0.61889};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{0, 4, 4, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);
        exp = "shape: (3, 5)\n" +
                "Non-zero data: [ 0.23804  0.38857  0.94397  0.61889 ]\n" +
                "Row Indices: [0, 0, 1, 2]\n" +
                "Col Indices: [0, 4, 4, 1]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.52615, 0.5363, 0.51364, 0.25336};
        aRowIndices = new int[]{0, 0, 2, 2};
        aColIndices = new int[]{1, 3, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);
        exp = "shape: (3, 5)\n" +
                "Non-zero data: [ 0.52615  0.5363  0.51364  0.25336 ]\n" +
                "Row Indices: [0, 0, 2, 2]\n" +
                "Col Indices: [1, 3, 1, 2]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.27709, 0.88769, 0.5211, 0.37339};
        aRowIndices = new int[]{0, 0, 2, 2};
        aColIndices = new int[]{1, 4, 0, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);
        exp = "shape: (3, 5)\n" +
                "Non-zero data: [ 0.27709  0.88769  0.5211  0.37339 ]\n" +
                "Row Indices: [0, 0, 2, 2]\n" +
                "Col Indices: [1, 4, 0, 4]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.29006, 0.13548, 0.05222, 0.94335};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{2, 4, 3, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);
        exp = "shape: (3, 5)\n" +
                "Non-zero data: [ 0.29006  0.13548  0.05222  0.94335 ]\n" +
                "Row Indices: [0, 0, 1, 2]\n" +
                "Col Indices: [2, 4, 3, 2]";

        assertEquals(exp, a.toString());
    }
}
