package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SparseVectorConversionTests {

    static int[] aIndices;
    static double[] aEntries;
    static int sparseSize;
    static SparseVector a;


    @BeforeAll
    static void setup() {
        aEntries = new double[]{1.345, -989.234, 5.15, 617.4};
        aIndices = new int[]{4, 56, 9903, 14643};
        sparseSize = 24023;
        a = new SparseVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void toMatrixTest() {
        double[] expEntries;
        int[][] expIndices;
        Shape expShape;
        SparseMatrix exp;

        // ------------------- Sub-case 1 -------------------
        expEntries = new double[]{1.345, -989.234, 5.15, 617.4};
        expIndices = new int[][]{{4, 56, 9903, 14643}, {0, 0, 0, 0}};
        expShape = new Shape(sparseSize, 1);
        exp = new SparseMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix());

        // ------------------- Sub-case 2 -------------------
        expEntries = new double[]{1.345, -989.234, 5.15, 617.4};
        expIndices = new int[][]{{4, 56, 9903, 14643}, {0, 0, 0, 0}};
        expShape = new Shape(sparseSize, 1);
        exp = new SparseMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix(true));

        // ------------------- Sub-case 3 -------------------
        expEntries = new double[]{1.345, -989.234, 5.15, 617.4};
        expIndices = new int[][]{{0, 0, 0, 0}, {4, 56, 9903, 14643}};
        expShape = new Shape(1, sparseSize);
        exp = new SparseMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix(false));
    }


    @Test
    void toComplex() {
        CNumber[] expEntries;
        int[] expIndices;
        int expSize;
        SparseCVector exp;

        // ------------------- Sub-case 1 -------------------
        expEntries = new CNumber[]{
                new CNumber(1.345), new CNumber(-989.234),
                new CNumber(5.15), new CNumber(617.4)};
        expIndices = new int[]{4, 56, 9903, 14643};
        expSize = sparseSize;
        exp = new SparseCVector(expSize, expEntries, expIndices);

        assertEquals(exp, a.toComplex());
    }


    @Test
    void toTensor() {
        double[] expEntries;
        int[][] expIndices;
        Shape expShape;
        SparseTensor exp;

        // ------------------- Sub-case 1 -------------------
        expEntries = new double[]{1.345, -989.234, 5.15, 617.4};
        expIndices = new int[][]{{4}, {56}, {9903}, {14643}};
        expShape = new Shape(sparseSize);
        exp = new SparseTensor(expShape, expEntries, expIndices);

        assertEquals(exp, a.toTensor());
    }


    @Test
    void toDenseTest() {
        double[] expEntries;
        Vector exp;

        // ------------------- Sub-case 1 -------------------
        expEntries = new double[sparseSize];
        expEntries[aIndices[0]] = aEntries[0];
        expEntries[aIndices[1]] = aEntries[1];
        expEntries[aIndices[2]] = aEntries[2];
        expEntries[aIndices[3]] = aEntries[3];
        exp = new Vector(expEntries);

        assertEquals(exp, a.toDense());
    }


    @Test
    void fromDenseTest() {
        double[] denseEntries;
        Vector denseVector;

        // ------------------- Sub-case 1 -------------------
        denseEntries = new double[sparseSize];
        denseEntries[aIndices[0]] = aEntries[0];
        denseEntries[aIndices[1]] = aEntries[1];
        denseEntries[aIndices[2]] = aEntries[2];
        denseEntries[aIndices[3]] = aEntries[3];
        denseVector = new Vector(denseEntries);

        assertEquals(a, SparseVector.fromDense(denseVector));
    }
}
