package com.flag4j.operations;

import static com.flag4j.operations.RealDenseTranspose.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealDenseTransposeTests {

    double[] A;
    double[] expTranspose;
    int numRows, numCols;


    @Test
    void standardTest() {
        // ------------- Sub-case 1 ---------------
        numRows = 3;
        numCols = 4;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));


        // ------------- Sub-case 2 ---------------
        numRows = 6;
        numCols = 2;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 12;
        numCols = 1;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));

        // ------------- Sub-case 3 ---------------
        numRows = 3;
        numCols = 3;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        expTranspose = new double[]{1, 4, 7, 2, 5, 8, 3, 6, 9};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new double[]{1.13};
        expTranspose = new double[]{1.13};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));
    }


    @Test
    void blockedTest() {
        // ------------- Sub-case 1 ---------------
        numRows = 3;
        numCols = 4;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));


        // ------------- Sub-case 2 ---------------
        numRows = 6;
        numCols = 2;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 12;
        numCols = 1;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));

        // ------------- Sub-case 3 ---------------
        numRows = 3;
        numCols = 3;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        expTranspose = new double[]{1, 4, 7, 2, 5, 8, 3, 6, 9};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new double[]{1.13};
        expTranspose = new double[]{1.13};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));
    }


    @Test
    void standardConcurrentTest() {
        // ------------- Sub-case 1 ---------------
        numRows = 3;
        numCols = 4;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));


        // ------------- Sub-case 2 ---------------
        numRows = 6;
        numCols = 2;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 12;
        numCols = 1;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));

        // ------------- Sub-case 3 ---------------
        numRows = 3;
        numCols = 3;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        expTranspose = new double[]{1, 4, 7, 2, 5, 8, 3, 6, 9};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new double[]{1.13};
        expTranspose = new double[]{1.13};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));
    }


    @Test
    void blockedConcurrentTest() {
        // ------------- Sub-case 1 ---------------
        numRows = 3;
        numCols = 4;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));


        // ------------- Sub-case 2 ---------------
        numRows = 6;
        numCols = 2;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 12;
        numCols = 1;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));

        // ------------- Sub-case 3 ---------------
        numRows = 3;
        numCols = 3;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        expTranspose = new double[]{1, 4, 7, 2, 5, 8, 3, 6, 9};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));


        // ------------- Sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new double[]{1.13};
        expTranspose = new double[]{1.13};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));
    }
}
