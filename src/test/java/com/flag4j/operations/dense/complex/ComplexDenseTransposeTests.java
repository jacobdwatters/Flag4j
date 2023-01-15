package com.flag4j.operations.dense.complex;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ComplexDenseTransposeTests {
    CNumber[] A;
    CNumber[] expTranspose, expTransposeH;
    int numRows, numCols;


    @Test
    void transposeTest() {
        // ------------- Sub-case 1 ---------------
        numRows = 3;
        numCols = 4;
        A = new CNumber[]{
                new CNumber(1), new CNumber(2), new CNumber(3), 
                new CNumber(4), new CNumber(5), new CNumber(6), 
                new CNumber(7), new CNumber(8), new CNumber(9), 
                new CNumber(10), new CNumber(11), new CNumber(12)
        };
        
        expTranspose = new CNumber[]{
                new CNumber(1), new CNumber(5), new CNumber(9), 
                new CNumber(2), new CNumber(6), new CNumber(10), 
                new CNumber(3), new CNumber(7), new CNumber(11), 
                new CNumber(4), new CNumber(8), new CNumber(12)};
        expTransposeH = new CNumber[]{
                new CNumber(1).conj(), new CNumber(5).conj(), new CNumber(9).conj(),
                new CNumber(2).conj(), new CNumber(6).conj(), new CNumber(10).conj(),
                new CNumber(3).conj(), new CNumber(7).conj(), new CNumber(11).conj(),
                new CNumber(4).conj(), new CNumber(8).conj(), new CNumber(12).conj()};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));
        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));

        assertArrayEquals(expTransposeH, standardMatrixHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, standardMatrixConcurrentHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixConcurrentHerm(A, numRows, numCols));

        // ------------- Sub-case 2 ---------------
        numRows = 6;
        numCols = 2;
        A = new CNumber[]{new CNumber(1), new CNumber(2), new CNumber(3), 
                new CNumber(4), new CNumber(5), new CNumber(6), 
                new CNumber(7), new CNumber(8), new CNumber(9), 
                new CNumber(10), new CNumber(11), new CNumber(12)};
        expTranspose = new CNumber[]{
                new CNumber(1), new CNumber(3), new CNumber(5), 
                new CNumber(7), new CNumber(9), new CNumber(11), 
                new CNumber(2), new CNumber(4), new CNumber(6), 
                new CNumber(8), new CNumber(10), new CNumber(12)};
        expTransposeH = new CNumber[]{
                new CNumber(1).conj(), new CNumber(3).conj(), new CNumber(5).conj(),
                new CNumber(7).conj(), new CNumber(9).conj(), new CNumber(11).conj(),
                new CNumber(2).conj(), new CNumber(4).conj(), new CNumber(6).conj(),
                new CNumber(8).conj(), new CNumber(10).conj(), new CNumber(12).conj()};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));
        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));

        assertArrayEquals(expTransposeH, standardMatrixHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, standardMatrixConcurrentHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixConcurrentHerm(A, numRows, numCols));

        // ------------- Sub-case 3 ---------------
        numRows = 12;
        numCols = 1;
        A = new CNumber[]{new CNumber(1), new CNumber(2), new CNumber(3),
                new CNumber(4), new CNumber(5), new CNumber(6),
                new CNumber(7), new CNumber(8), new CNumber(9),
                new CNumber(10), new CNumber(11), new CNumber(12)};
        expTranspose = new CNumber[]{
                new CNumber(1), new CNumber(2), new CNumber(3),
                new CNumber(4), new CNumber(5), new CNumber(6),
                new CNumber(7), new CNumber(8), new CNumber(9), 
                new CNumber(10), new CNumber(11), new CNumber(12)};
        expTransposeH = new CNumber[]{
                new CNumber(1).conj(), new CNumber(2).conj(), new CNumber(3).conj(),
                new CNumber(4).conj(), new CNumber(5).conj(), new CNumber(6).conj(),
                new CNumber(7).conj(), new CNumber(8).conj(), new CNumber(9).conj(),
                new CNumber(10).conj(), new CNumber(11).conj(), new CNumber(12).conj()};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));
        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));

        assertArrayEquals(expTransposeH, standardMatrixHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, standardMatrixConcurrentHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixConcurrentHerm(A, numRows, numCols));

        // ------------- Sub-case 3 ---------------
        numRows = 3;
        numCols = 3;
        A = new CNumber[]{
                new CNumber(1), new CNumber(2), new CNumber(3), 
                new CNumber(4), new CNumber(5), new CNumber(6), 
                new CNumber(7), new CNumber(8), new CNumber(9)};
        expTranspose = new CNumber[]{
                new CNumber(1), new CNumber(4), new CNumber(7), 
                new CNumber(2), new CNumber(5), new CNumber(8),
                new CNumber(3), new CNumber(6), new CNumber(9)};
        expTransposeH = new CNumber[]{
                new CNumber(1).conj(), new CNumber(4).conj(), new CNumber(7).conj(),
                new CNumber(2).conj(), new CNumber(5).conj(), new CNumber(8).conj(),
                new CNumber(3).conj(), new CNumber(6).conj(), new CNumber(9).conj()};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));
        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));

        assertArrayEquals(expTransposeH, standardMatrixHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, standardMatrixConcurrentHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixConcurrentHerm(A, numRows, numCols));

        // ------------- Sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new CNumber[]{new CNumber(1.13)};
        expTranspose = new CNumber[]{new CNumber(1.13)};
        expTransposeH = new CNumber[]{new CNumber(1.13).conj()};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));
        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));

        assertArrayEquals(expTransposeH, standardMatrixHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, standardMatrixConcurrentHerm(A, numRows, numCols));
        assertArrayEquals(expTranspose, blockedMatrixConcurrentHerm(A, numRows, numCols));
    }
}
