package com.flag4j.sparse_complex_vector;


import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooCVectorConversionTests {

    static int[] aIndices;
    static CNumber[] aEntries;
    static int sparseSize;
    static CooCVector a;


    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{
                new CNumber(2.455, -83.6), new CNumber(0, 24.56),
                new CNumber(24.56), new CNumber(-9356.1, 35)
        };
        aIndices = new int[]{4, 56, 9903, 14643};
        sparseSize = 24_023;
        a = new CooCVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void toMatrixTestCase() {
        CNumber[] expEntries;
        int[][] expIndices;
        Shape expShape;
        CooCMatrix exp;

        // ------------------- Sub-case 1 -------------------
        expEntries = new CNumber[]{
                new CNumber(2.455, -83.6), new CNumber(0, 24.56),
                new CNumber(24.56), new CNumber(-9356.1, 35)
        };
        expIndices = new int[][]{{4, 56, 9903, 14643}, {0, 0, 0, 0}};
        expShape = new Shape(sparseSize, 1);
        exp = new CooCMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix());

        // ------------------- Sub-case 2 -------------------
        expEntries = new CNumber[]{
                new CNumber(2.455, -83.6), new CNumber(0, 24.56),
                new CNumber(24.56), new CNumber(-9356.1, 35)
        };
        expIndices = new int[][]{{4, 56, 9903, 14643}, {0, 0, 0, 0}};
        expShape = new Shape(sparseSize, 1);
        exp = new CooCMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix(true));

        // ------------------- Sub-case 3 -------------------
        expEntries = new CNumber[]{
                new CNumber(2.455, -83.6), new CNumber(0, 24.56),
                new CNumber(24.56), new CNumber(-9356.1, 35)
        };
        expIndices = new int[][]{{0, 0, 0, 0}, {4, 56, 9903, 14643}};
        expShape = new Shape(1, sparseSize);
        exp = new CooCMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix(false));
    }


    @Test
    void toRealTestCase() {
        double[] expEntries;
        int[] expIndices;
        int expSize;
        CooVector exp;

        // ------------------- Sub-case 1 -------------------
        expEntries = new double[]{2.455, 0, 24.56, -9356.1};
        expIndices = new int[]{4, 56, 9903, 14643};
        expSize = sparseSize;
        exp = new CooVector(expSize, expEntries, expIndices);

        assertEquals(exp, a.toReal());
    }


    @Test
    void toTensor() {
        CNumber[] expEntries;
        int[][] expIndices;
        Shape expShape;
        CooCTensor exp;

        // ------------------- Sub-case 1 -------------------
        expEntries = new CNumber[]{
                new CNumber(2.455, -83.6), new CNumber(0, 24.56),
                new CNumber(24.56), new CNumber(-9356.1, 35)
        };
        expIndices = new int[][]{{4}, {56}, {9903}, {14643}};
        expShape = new Shape(sparseSize);
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, a.toTensor());
    }


    @Test
    void toDenseTestCase() {
        CNumber[] expEntries;
        CVector exp;

        // ------------------- Sub-case 1 -------------------
        expEntries = new CNumber[sparseSize];
        expEntries[aIndices[0]] = aEntries[0].copy();
        expEntries[aIndices[1]] = aEntries[1].copy();
        expEntries[aIndices[2]] = aEntries[2].copy();
        expEntries[aIndices[3]] = aEntries[3].copy();
        exp = new CVector(expEntries);

        assertEquals(exp, a.toDense());
    }


    @Test
    void fromDenseTestCase() {
        CNumber[] denseEntries;
        CVector denseVector;

        // ------------------- Sub-case 1 -------------------
        denseEntries = new CNumber[sparseSize];
        ArrayUtils.fillZeros(denseEntries);
        denseEntries[aIndices[0]] = aEntries[0].copy();
        denseEntries[aIndices[1]] = aEntries[1].copy();
        denseEntries[aIndices[2]] = aEntries[2].copy();
        denseEntries[aIndices[3]] = aEntries[3].copy();
        denseVector = new CVector(denseEntries);

        assertEquals(a, CooCVector.fromDense(denseVector));
    }
}
