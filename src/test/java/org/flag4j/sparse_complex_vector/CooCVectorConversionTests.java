package org.flag4j.sparse_complex_vector;


import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CooCTensor;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
        CVectorOld exp;

        // ------------------- Sub-case 1 -------------------
        expEntries = new CNumber[sparseSize];
        Arrays.fill(expEntries, CNumber.ZERO);       
        expEntries[aIndices[0]] = aEntries[0];
        expEntries[aIndices[1]] = aEntries[1];
        expEntries[aIndices[2]] = aEntries[2];
        expEntries[aIndices[3]] = aEntries[3];
        exp = new CVectorOld(expEntries);

        assertTrue(exp.tensorEquals(a.toDense()));
    }


    @Test
    void fromDenseTestCase() {
        CNumber[] denseEntries;
        CVectorOld denseVector;

        // ------------------- Sub-case 1 -------------------
        denseEntries = new CNumber[sparseSize];
        Arrays.fill(denseEntries, CNumber.ZERO);
        denseEntries[aIndices[0]] = aEntries[0];
        denseEntries[aIndices[1]] = aEntries[1];
        denseEntries[aIndices[2]] = aEntries[2];
        denseEntries[aIndices[3]] = aEntries[3];
        denseVector = new CVectorOld(denseEntries);

        assertEquals(a, CooCVector.fromDense(denseVector));
    }
}
