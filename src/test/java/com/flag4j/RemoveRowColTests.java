package com.flag4j;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class RemoveRowColTests {
    double[][] aEntries, expEntries;
    Matrix A, exp;
    int index;

    @Test
    void removeRowTest() {
        // ------------ Sub-case 1 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 1;
        expEntries = new double[][]{{1, 2, 3}, {7, 8, 9}, {10, 11, 12}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeRow(index));

        // ------------ Sub-case 2 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 3;
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeRow(index));

        // ------------ Sub-case 3 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = -1;
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeRow(index));

        // ------------ Sub-case 4 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 5;
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeRow(index));
    }


    @Test
    void removeColTest() {
        // ------------ Sub-case 1 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 1;
        expEntries = new double[][]{{1, 3}, {4, 6}, {7, 9}, {10, 12}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeCol(index));

        // ------------ Sub-case 2 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 2;
        expEntries = new double[][]{{1, 2}, {4, 5}, {7, 8}, {10, 11}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeCol(index));

        // ------------ Sub-case 3 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = -1;
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeCol(index));

        // ------------ Sub-case 4 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 5;
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeCol(index));
    }
}
