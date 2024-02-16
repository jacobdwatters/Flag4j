package com.flag4j.sparse_matrix;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.Shape;
import com.flag4j.dense.CVector;
import com.flag4j.dense.Vector;
import com.flag4j.sparse.CooCMatrix;
import com.flag4j.sparse.CooCVector;
import com.flag4j.sparse.CooMatrix;
import com.flag4j.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CooMatrixStackVectorTests {

    @Test
    void realSparseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bindices;
        double[] bEntries;
        CooVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.1144, 0.74731, 0.51996, 0.09408, 0.20219};
        aRowIndices = new int[]{0, 1, 1, 2, 2};
        aColIndices = new int[]{4, 1, 4, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new double[]{0.51982, 0.38743};
        bindices = new int[]{1, 3};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(4, 5);
        expEntries = new double[]{0.1144, 0.74731, 0.51996, 0.09408, 0.20219, 0.51982, 0.38743};
        expRowIndices = new int[]{0, 1, 1, 2, 2, 3, 3};
        expColIndices = new int[]{4, 1, 4, 3, 4, 1, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{0.3056};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new double[]{0.67691};
        bindices = new int[]{0};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new double[]{0.3056, 0.67691};
        expRowIndices = new int[]{0, 1};
        expColIndices = new int[]{0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.40429, 0.854, 0.6952};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{2, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.31166};
        bindices = new int[]{1};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        CooMatrix finala = a;
        CooVector finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void complexSparseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bindices;
        CNumber[] bEntries;
        CooCVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.50192, 0.42775, 0.82651, 0.77339, 0.54693};
        aRowIndices = new int[]{0, 0, 0, 1, 2};
        aColIndices = new int[]{2, 3, 4, 4, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new CNumber[]{new CNumber("0.6745+0.85839i"), new CNumber("0.50328+0.7547i")};
        bindices = new int[]{0, 2};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(4, 5);
        expEntries = new CNumber[]{new CNumber("0.50192"), new CNumber("0.42775"), new CNumber("0.82651"), new CNumber("0.77339"), new CNumber("0.54693"), new CNumber("0.6745+0.85839i"), new CNumber("0.50328+0.7547i")};
        expRowIndices = new int[]{0, 0, 0, 1, 2, 3, 3};
        expColIndices = new int[]{2, 3, 4, 4, 0, 0, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{0.97237};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new CNumber[]{new CNumber("0.70721+0.27207i")};
        bindices = new int[]{1};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.97237"), new CNumber("0.70721+0.27207i")};
        expRowIndices = new int[]{0, 1};
        expColIndices = new int[]{1, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.8983, 0.21095, 0.83482};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{1, 1, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new CNumber[]{new CNumber("0.57138+0.88361i")};
        bindices = new int[]{0};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        CooMatrix finala = a;
        CooCVector finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void realDenseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double[] bEntries;
        Vector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.98534, 0.23116, 0.96232, 0.80171, 0.03408};
        aRowIndices = new int[]{0, 0, 1, 2, 2};
        aColIndices = new int[]{1, 3, 0, 0, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.46319, 0.28211, 0.05495, 0.42074, 0.55151};
        b = new Vector(bEntries);

        expShape = new Shape(4, 5);
        expEntries = new double[]{0.98534, 0.23116, 0.96232, 0.80171, 0.03408, 0.46319, 0.28211, 0.05495, 0.42074, 0.55151};
        expRowIndices = new int[]{0, 0, 1, 2, 2, 3, 3, 3, 3, 3};
        expColIndices = new int[]{1, 3, 0, 0, 3, 0, 1, 2, 3, 4};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{0.86177};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.07394, 0.00648};
        b = new Vector(bEntries);

        expShape = new Shape(2, 2);
        expEntries = new double[]{0.86177, 0.07394, 0.00648};
        expRowIndices = new int[]{0, 1, 1};
        expColIndices = new int[]{1, 0, 1};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.52439, 0.40097, 0.06923};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{1, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.52202, 0.27861, 0.28872};
        b = new Vector(bEntries);

        CooMatrix finala = a;
        Vector finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void complexDenseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        CNumber[] bEntries;
        CVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.21625, 0.33009, 0.1306, 0.42002, 0.57863};
        aRowIndices = new int[]{0, 0, 0, 1, 1};
        aColIndices = new int[]{0, 1, 4, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.48822+0.82255i"), new CNumber("0.22266+0.42225i"), new CNumber("0.0867+0.09891i"), new CNumber("0.04798+0.2574i"), new CNumber("0.27218+0.28588i")};
        b = new CVector(bEntries);

        expShape = new Shape(4, 5);
        expEntries = new CNumber[]{new CNumber("0.21625"), new CNumber("0.33009"), new CNumber("0.1306"), new CNumber("0.42002"), new CNumber("0.57863"), new CNumber("0.48822+0.82255i"), new CNumber("0.22266+0.42225i"), new CNumber("0.0867+0.09891i"), new CNumber("0.04798+0.2574i"), new CNumber("0.27218+0.28588i")};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 3, 3, 3, 3, 3};
        expColIndices = new int[]{0, 1, 4, 0, 1, 0, 1, 2, 3, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{0.38689};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.6287+0.83629i"), new CNumber("0.90518+0.09729i")};
        b = new CVector(bEntries);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.38689"), new CNumber("0.6287+0.83629i"), new CNumber("0.90518+0.09729i")};
        expRowIndices = new int[]{0, 1, 1};
        expColIndices = new int[]{0, 0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.88601, 0.64124, 0.04454};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{1, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.67101+0.86109i"), new CNumber("0.88748+0.91107i"), new CNumber("0.7637+0.52437i")};
        b = new CVector(bEntries);

        CooMatrix finala = a;
        CVector finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }
}
