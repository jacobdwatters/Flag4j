package org.flag4j.complex_matrix;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CTensorOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixConversionTests {

    CNumber[] expEntries;
    CTensorOld expTensor, tensor;
    CVectorOld expVector, vector;
    Shape expTensorShape;

    CNumber[][] aEntries;
    double[][] expRealEntries;
    CMatrixOld A;
    MatrixOld expReal, real;

    @Test
    void toRealTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(1, 7.233), new CNumber(-0.34436, 13)},
                {new CNumber(80.234, -9), new CNumber(Double.POSITIVE_INFINITY, 885.224)},
                {new CNumber(843.15), new CNumber(99.3434, 146)}};
        A = new CMatrixOld(aEntries);
        expRealEntries = new double[][]{
                {1, -0.34436},
                {80.234, Double.POSITIVE_INFINITY},
                {843.15, 99.3434}};
        expReal = new MatrixOld(expRealEntries);

        real = A.toReal();

        assertEquals(expReal, real);
    }


    @Test
    void toTensorTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(1, 7.233), new CNumber(-0.34436, 13)},
                {new CNumber(80.234, -9), new CNumber(Double.POSITIVE_INFINITY, 885.224)},
                {new CNumber(843.15), new CNumber(99.3434, 146)}};
        A = new CMatrixOld(aEntries);
        expEntries = new CNumber[]{
                new CNumber(1, 7.233), new CNumber(-0.34436, 13),
                new CNumber(80.234, -9), new CNumber(Double.POSITIVE_INFINITY, 885.224),
                new CNumber(843.15), new CNumber(99.3434, 146)};
        expTensorShape = new Shape(A.numRows, A.numCols);
        expTensor = new CTensorOld(expTensorShape, expEntries);

        tensor = A.toTensor();
        assertArrayEquals(A.entries, tensor.entries);
        assertEquals(A.shape, tensor.shape);
        assertEquals(2, tensor.getRank());
    }


    @Test
    void toVectorTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(1, 7.233), new CNumber(-0.34436, 13)},
                {new CNumber(80.234, -9), new CNumber(Double.POSITIVE_INFINITY, 885.224)},
                {new CNumber(843.15), new CNumber(99.3434, 146)}};
        A = new CMatrixOld(aEntries);
        expEntries = new CNumber[]{
                new CNumber(1, 7.233), new CNumber(-0.34436, 13),
                new CNumber(80.234, -9), new CNumber(Double.POSITIVE_INFINITY, 885.224),
                new CNumber(843.15), new CNumber(99.3434, 146)};
        expVector = new CVectorOld(expEntries);

        vector = A.toVector();
        assertArrayEquals(expVector.entries, vector.entries);
    }
}
