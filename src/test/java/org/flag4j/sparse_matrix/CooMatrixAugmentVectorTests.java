package org.flag4j.sparse_matrix;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixAugmentVectorTests {

    @Test
    void realSparseAugmentTest() {
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
        aEntries = new double[]{0.33367, 0.26667, 0.24386, 0.05929, 0.90991};
        aRowIndices = new int[]{0, 0, 0, 1, 2};
        aColIndices = new int[]{1, 2, 4, 0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.74357};
        bindices = new int[]{2};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(3, 6);
        expEntries = new double[]{0.33367, 0.26667, 0.24386, 0.05929, 0.90991, 0.74357};
        expRowIndices = new int[]{0, 0, 0, 1, 2, 2};
        expColIndices = new int[]{1, 2, 4, 0, 0, 5};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new double[]{0.48537};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new double[]{0.18184};
        bindices = new int[]{0};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new double[]{0.48537, 0.18184};
        expRowIndices = new int[]{0, 0};
        expColIndices = new int[]{0, 1};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.39748, 0.85537, 0.1253};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{3, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.10397};
        bindices = new int[]{2};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        CooMatrix finala = a;
        CooVector finalb = b;
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
        aEntries = new double[]{0.86322, 0.88853, 0.48177, 0.47086, 0.10662};
        aRowIndices = new int[]{0, 1, 1, 1, 2};
        aColIndices = new int[]{4, 0, 2, 4, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new CNumber[]{new CNumber("0.38636+0.61929i")};
        bindices = new int[]{0};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(3, 6);
        expEntries = new CNumber[]{new CNumber("0.86322"), new CNumber("0.38636+0.61929i"), new CNumber("0.88853"), new CNumber("0.48177"), new CNumber("0.47086"), new CNumber("0.10662")};
        expRowIndices = new int[]{0, 0, 1, 1, 1, 2};
        expColIndices = new int[]{4, 5, 0, 2, 4, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new double[]{0.42257};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new CNumber[]{new CNumber("0.59203+0.46515i")};
        bindices = new int[]{1};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.42257"), new CNumber("0.59203+0.46515i")};
        expRowIndices = new int[]{1, 1};
        expColIndices = new int[]{0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.86643, 0.39913, 0.09523};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{0, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new CNumber[]{new CNumber("0.2738+0.92376i")};
        bindices = new int[]{0};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        CooMatrix finala = a;
        CooCVector finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }


    @Test
    void realDenseAugmentTest() {
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
        aEntries = new double[]{0.33461, 0.2762, 0.18055, 0.15661, 0.99496};
        aRowIndices = new int[]{0, 0, 2, 2, 2};
        aColIndices = new int[]{2, 4, 0, 2, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.96807, 0.03181, 0.52901};
        b = new Vector(bEntries);

        expShape = new Shape(3, 6);
        expEntries = new double[]{0.33461, 0.2762, 0.96807, 0.03181, 0.18055, 0.15661, 0.99496, 0.52901};
        expRowIndices = new int[]{0, 0, 0, 1, 2, 2, 2, 2};
        expColIndices = new int[]{2, 4, 5, 5, 0, 2, 3, 5};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new double[]{0.27644};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.94279, 0.34055};
        b = new Vector(bEntries);

        expShape = new Shape(2, 2);
        expEntries = new double[]{0.94279, 0.27644, 0.34055};
        expRowIndices = new int[]{0, 1, 1};
        expColIndices = new int[]{1, 0, 1};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.29305, 0.13286, 0.14255};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{3, 1, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.169, 0.67177, 0.25641};
        b = new Vector(bEntries);

        CooMatrix finala = a;
        Vector finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }


    @Test
    void complexDenseAugmentTest() {
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
        aEntries = new double[]{0.3522, 0.06776, 0.05492, 0.19119, 0.87698};
        aRowIndices = new int[]{0, 0, 1, 1, 2};
        aColIndices = new int[]{1, 3, 2, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.80314+0.76471i"), new CNumber("0.61237+0.67766i"), new CNumber("0.43664+0.52055i")};
        b = new CVector(bEntries);

        expShape = new Shape(3, 6);
        expEntries = new CNumber[]{new CNumber("0.3522"), new CNumber("0.06776"), new CNumber("0.80314+0.76471i"), new CNumber("0.05492"), new CNumber("0.19119"), new CNumber("0.61237+0.67766i"), new CNumber("0.87698"), new CNumber("0.43664+0.52055i")};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 1, 2, 2};
        expColIndices = new int[]{1, 3, 5, 2, 3, 5, 4, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new double[]{0.46871};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.84346+0.39347i"), new CNumber("0.67741+0.3733i")};
        b = new CVector(bEntries);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.84346+0.39347i"), new CNumber("0.46871"), new CNumber("0.67741+0.3733i")};
        expRowIndices = new int[]{0, 1, 1};
        expColIndices = new int[]{1, 0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.41925, 0.0637, 0.23033};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{2, 3, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.42232+0.32179i"), new CNumber("0.06046+0.8805i"), new CNumber("0.59928+0.29377i")};
        b = new CVector(bEntries);

        CooMatrix finala = a;
        CVector finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }
}
