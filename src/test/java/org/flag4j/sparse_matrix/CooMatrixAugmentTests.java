package org.flag4j.sparse_matrix;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.Matrix;
import org.flag4j.sparse.CooCMatrix;
import org.flag4j.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixAugmentTests {

    @Test
    void realSparseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        double[] bEntries;
        CooMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.93474, 0.28545, 0.33318};
        aRowIndices = new int[]{1, 2, 2};
        aColIndices = new int[]{2, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 2);
        bEntries = new double[]{0.2452};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 7);
        expEntries = new double[]{0.2452, 0.93474, 0.28545, 0.33318};
        expRowIndices = new int[]{0, 1, 2, 2};
        expColIndices = new int[]{5, 2, 2, 4};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new double[]{0.05933};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{2};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(2, 4);
        expEntries = new double[]{0.05933};
        expRowIndices = new int[]{0};
        expColIndices = new int[]{3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.74244, 0.32181};
        aRowIndices = new int[]{0, 0};
        aColIndices = new int[]{2, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 4);
        bEntries = new double[]{0.38007, 0.3302};
        bRowIndices = new int[]{1, 2};
        bColIndices = new int[]{0, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix finala = a;
        CooMatrix finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }


    @Test
    void complexSparseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        CooCMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.09732, 0.12299, 0.80847};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{3, 4, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 2);
        bEntries = new CNumber[]{new CNumber("0.72792+0.70995i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 7);
        expEntries = new CNumber[]{new CNumber("0.09732"), new CNumber("0.72792+0.70995i"), new CNumber("0.12299"), new CNumber("0.80847")};
        expRowIndices = new int[]{0, 0, 1, 2};
        expColIndices = new int[]{3, 6, 4, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new CNumber[]{new CNumber("0.18942+0.29i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(2, 4);
        expEntries = new CNumber[]{new CNumber("0.18942+0.29i")};
        expRowIndices = new int[]{0};
        expColIndices = new int[]{1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.00967, 0.832};
        aRowIndices = new int[]{0, 1};
        aColIndices = new int[]{2, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 4);
        bEntries = new CNumber[]{new CNumber("0.83588+0.82531i"), new CNumber("0.90652+0.03017i")};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{0, 1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix finala = a;
        CooCMatrix finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }


    @Test
    void realDenseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double[][] bEntries;
        Matrix b;

        double[][] expEntries;
        Matrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.24209, 0.05396, 0.55784};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{1, 4, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.45827, 0.98222},
                {0.10505, 0.01145},
                {0.44006, 0.32708}};
        b = new Matrix(bEntries);

        expEntries = new double[][]{
                {0.0, 0.24209, 0.0, 0.0, 0.05396, 0.45827, 0.98222},
                {0.0, 0.0, 0.0, 0.55784, 0.0, 0.10505, 0.01145},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.44006, 0.32708}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.17793, 0.69724, 0.30593},
                {0.57252, 0.42922, 0.76316}};
        b = new Matrix(bEntries);

        expEntries = new double[][]{
                {0.0, 0.17793, 0.69724, 0.30593},
                {0.0, 0.57252, 0.42922, 0.76316}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.82265, 0.39013};
        aRowIndices = new int[]{0, 1};
        aColIndices = new int[]{3, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.76158, 0.77484, 0.16097, 0.03449},
                {0.63107, 0.56961, 0.68014, 0.78149},
                {0.83588, 0.67375, 0.36044, 0.12535}};
        b = new Matrix(bEntries);

        CooMatrix finala = a;
        Matrix finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }


    @Test
    void complexDenseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        CNumber[][] bEntries;
        CMatrix b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.4513, 0.10819, 0.52671};
        aRowIndices = new int[]{0, 2, 2};
        aColIndices = new int[]{1, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.69748+0.29275i"), new CNumber("0.30393+0.95144i")},
                {new CNumber("0.62313+0.50274i"), new CNumber("0.89225+0.42919i")},
                {new CNumber("0.30844+0.64805i"), new CNumber("0.20611+0.94464i")}};
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.4513"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.69748+0.29275i"), new CNumber("0.30393+0.95144i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.62313+0.50274i"), new CNumber("0.89225+0.42919i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.10819"), new CNumber("0.52671"), new CNumber("0.30844+0.64805i"), new CNumber("0.20611+0.94464i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.58223+0.21348i"), new CNumber("0.53616+0.63303i"), new CNumber("0.5126+0.48521i")},
                {new CNumber("0.39158+0.05234i"), new CNumber("0.80737+0.00379i"), new CNumber("0.21414+0.55187i")}};
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.58223+0.21348i"), new CNumber("0.53616+0.63303i"), new CNumber("0.5126+0.48521i")},
                {new CNumber("0.0"), new CNumber("0.39158+0.05234i"), new CNumber("0.80737+0.00379i"), new CNumber("0.21414+0.55187i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.47074, 0.8649};
        aRowIndices = new int[]{0, 1};
        aColIndices = new int[]{1, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.72123+0.46784i"), new CNumber("0.80452+0.80951i"), new CNumber("0.02527+0.50829i"), new CNumber("0.69448+0.13451i")},
                {new CNumber("0.99776+0.58328i"), new CNumber("0.84592+0.85895i"), new CNumber("0.16925+0.12029i"), new CNumber("0.04582+0.80704i")},
                {new CNumber("0.25605+0.00949i"), new CNumber("0.79089+0.73571i"), new CNumber("0.36083+0.2747i"), new CNumber("0.3623+0.57842i")}};
        b = new CMatrix(bEntries);

        CooMatrix finala = a;
        CMatrix finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }
}
