package com.flag4j.complex_sparse_matrix;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.Shape;
import com.flag4j.dense.CVector;
import com.flag4j.dense.Vector;
import com.flag4j.sparse.CooCMatrix;
import com.flag4j.sparse.CooCVector;
import com.flag4j.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixStackVectorsTests {

    @Test
    void realSparseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bindices;
        double[] bEntries;
        CooVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.95935+0.369i"), new CNumber("0.43672+0.52664i"), new CNumber("0.87388+0.76344i"), new CNumber("0.31699+0.21873i"), new CNumber("0.74985+0.07535i")};
        aRowIndices = new int[]{0, 0, 1, 1, 1};
        aColIndices = new int[]{2, 3, 0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new double[]{0.68902, 0.30089};
        bindices = new int[]{0, 3};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(4, 5);
        expEntries = new CNumber[]{new CNumber("0.95935+0.369i"), new CNumber("0.43672+0.52664i"), new CNumber("0.87388+0.76344i"), new CNumber("0.31699+0.21873i"), new CNumber("0.74985+0.07535i"), new CNumber("0.68902"), new CNumber("0.30089")};
        expRowIndices = new int[]{0, 0, 1, 1, 1, 3, 3};
        expColIndices = new int[]{2, 3, 0, 1, 2, 0, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{new CNumber("0.42054+0.78554i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new double[]{0.72719};
        bindices = new int[]{0};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.42054+0.78554i"), new CNumber("0.72719")};
        expRowIndices = new int[]{0, 1};
        expColIndices = new int[]{0, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new CNumber[]{new CNumber("0.28454+0.39471i"), new CNumber("0.45388+0.07664i"), new CNumber("0.4625+0.33759i")};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{2, 0, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.12834};
        bindices = new int[]{0};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        CooCMatrix finala = a;
        CooVector finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void complexSparseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

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
        aEntries = new CNumber[]{new CNumber("0.57405+0.661i"), new CNumber("0.98971+0.31897i"), new CNumber("0.42169+0.39332i"), new CNumber("0.94717+0.10069i"), new CNumber("0.20536+0.14784i")};
        aRowIndices = new int[]{1, 2, 2, 2, 2};
        aColIndices = new int[]{1, 0, 1, 3, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new CNumber[]{new CNumber("0.47162+0.56911i"), new CNumber("0.80047+0.49525i")};
        bindices = new int[]{0, 2};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(4, 5);
        expEntries = new CNumber[]{new CNumber("0.57405+0.661i"), new CNumber("0.98971+0.31897i"), new CNumber("0.42169+0.39332i"), new CNumber("0.94717+0.10069i"), new CNumber("0.20536+0.14784i"), new CNumber("0.47162+0.56911i"), new CNumber("0.80047+0.49525i")};
        expRowIndices = new int[]{1, 2, 2, 2, 2, 3, 3};
        expColIndices = new int[]{1, 0, 1, 3, 4, 0, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{new CNumber("0.92839+0.63971i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new CNumber[]{new CNumber("0.64474+0.85723i")};
        bindices = new int[]{0};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.92839+0.63971i"), new CNumber("0.64474+0.85723i")};
        expRowIndices = new int[]{0, 1};
        expColIndices = new int[]{1, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new CNumber[]{new CNumber("0.77989+0.92369i"), new CNumber("0.52864+0.69779i"), new CNumber("0.58062+0.71533i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{0, 2, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new CNumber[]{new CNumber("0.1783+0.79081i")};
        bindices = new int[]{2};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        CooCMatrix finala = a;
        CooCVector finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void realDenseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        double[] bEntries;
        Vector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.82346+0.5447i"), new CNumber("0.98536+0.00179i"), new CNumber("0.24957+0.43968i"), new CNumber("0.06434+0.2831i"), new CNumber("0.60887+0.53339i")};
        aRowIndices = new int[]{0, 0, 0, 1, 2};
        aColIndices = new int[]{0, 1, 4, 1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.05556, 0.67678, 0.9478, 0.41292, 0.67343};
        b = new Vector(bEntries);

        expShape = new Shape(4, 5);
        expEntries = new CNumber[]{new CNumber("0.82346+0.5447i"), new CNumber("0.98536+0.00179i"), new CNumber("0.24957+0.43968i"), new CNumber("0.06434+0.2831i"), new CNumber("0.60887+0.53339i"), new CNumber("0.05556"), new CNumber("0.67678"), new CNumber("0.9478"), new CNumber("0.41292"), new CNumber("0.67343")};
        expRowIndices = new int[]{0, 0, 0, 1, 2, 3, 3, 3, 3, 3};
        expColIndices = new int[]{0, 1, 4, 1, 1, 0, 1, 2, 3, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{new CNumber("0.22002+0.05773i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.59072, 0.60299};
        b = new Vector(bEntries);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.22002+0.05773i"), new CNumber("0.59072"), new CNumber("0.60299")};
        expRowIndices = new int[]{0, 1, 1};
        expColIndices = new int[]{1, 0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new CNumber[]{new CNumber("0.15127+0.44678i"), new CNumber("0.35932+0.80153i"), new CNumber("0.00951+0.14255i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{0, 1, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.90256, 0.08861, 0.49854};
        b = new Vector(bEntries);

        CooCMatrix finala = a;
        Vector finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void complexDenseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        CNumber[] bEntries;
        CVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.70739+0.52162i"), new CNumber("0.73024+0.28129i"), new CNumber("0.22347+0.6196i"), new CNumber("0.07587+0.34439i"), new CNumber("0.31589+0.36087i")};
        aRowIndices = new int[]{0, 0, 1, 2, 2};
        aColIndices = new int[]{1, 3, 2, 2, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.33633+0.96882i"), new CNumber("0.66443+0.10543i"), new CNumber("0.0555+0.19518i"), new CNumber("0.63482+0.63059i"), new CNumber("0.25934+0.31857i")};
        b = new CVector(bEntries);

        expShape = new Shape(4, 5);
        expEntries = new CNumber[]{new CNumber("0.70739+0.52162i"), new CNumber("0.73024+0.28129i"), new CNumber("0.22347+0.6196i"), new CNumber("0.07587+0.34439i"), new CNumber("0.31589+0.36087i"), new CNumber("0.33633+0.96882i"), new CNumber("0.66443+0.10543i"), new CNumber("0.0555+0.19518i"), new CNumber("0.63482+0.63059i"), new CNumber("0.25934+0.31857i")};
        expRowIndices = new int[]{0, 0, 1, 2, 2, 3, 3, 3, 3, 3};
        expColIndices = new int[]{1, 3, 2, 2, 3, 0, 1, 2, 3, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{new CNumber("0.93373+0.35044i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.66459+0.41638i"), new CNumber("0.26529+0.30038i")};
        b = new CVector(bEntries);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.93373+0.35044i"), new CNumber("0.66459+0.41638i"), new CNumber("0.26529+0.30038i")};
        expRowIndices = new int[]{0, 1, 1};
        expColIndices = new int[]{0, 0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new CNumber[]{new CNumber("0.83344+0.44261i"), new CNumber("0.79701+0.77615i"), new CNumber("0.01334+0.10945i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{2, 3, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.15167+0.14269i"), new CNumber("0.04675+0.07281i"), new CNumber("0.10411+0.4936i")};
        b = new CVector(bEntries);

        CooCMatrix finala = a;
        CVector finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }
}
