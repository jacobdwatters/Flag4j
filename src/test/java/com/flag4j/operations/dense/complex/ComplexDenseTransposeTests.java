package com.flag4j.operations.dense.complex;

import static com.flag4j.operations.dense.complex.ComplexDenseTranspose.*;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ComplexDenseTransposeTests {
    CNumber[] A;
    CNumber[] expTranspose;
    int numRows, numCols;


    @Test
    void standardTest() {
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

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));


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

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));


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

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));

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

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new CNumber[]{new CNumber(1.13)};
        expTranspose = new CNumber[]{new CNumber(1.13)};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));
    }


    @Test
    void blockedTest() {
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

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));


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

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));


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

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));

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

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new CNumber[]{new CNumber(1.13)};
        expTranspose = new CNumber[]{new CNumber(1.13)};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));
    }


    @Test
    void standardConcurrentTest() {
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

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));


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

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));


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

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));

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

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new CNumber[]{new CNumber(1.13)};
        expTranspose = new CNumber[]{new CNumber(1.13)};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));
    }


    @Test
    void blockedConcurrentTest() {
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

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));


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

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));


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

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));

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

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new CNumber[]{new CNumber(1.13)};
        expTranspose = new CNumber[]{new CNumber(1.13)};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));
    }
}
