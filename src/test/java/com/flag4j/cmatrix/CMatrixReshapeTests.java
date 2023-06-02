package com.flag4j.cmatrix;

import com.flag4j.CMatrix;
import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CMatrixReshapeTests {
    CNumber[] entries = {
            new CNumber(1, -9.345), new CNumber(2), new CNumber(-3.444, -4371.5),
            new CNumber(-4), new CNumber(5), new CNumber(-6.44),
            new CNumber(7.77, -72.4667), new CNumber(8.435, 43.1), new CNumber(9),
            new CNumber(10.4, 156), new CNumber(0, 11), new CNumber(12.2344)};
    CMatrix A = new CMatrix(new Shape(3, 4), entries);
    CMatrix B;
    Shape expShape;
    int rows, cols;


    @Test
    void reshapeTest() {
        // --------------- Sub-case 1 ---------------
        expShape = new Shape(4, 3);
        B = A.reshape(expShape.copy());
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-case 2 ---------------
        expShape = new Shape(1, 12);
        B = A.reshape(expShape.copy());
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-case 3 ---------------
        expShape = new Shape(2, 6);
        B = A.reshape(expShape.copy());
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-case 4 ---------------
        expShape = new Shape(6, 2);
        B = A.reshape(expShape.copy());
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-case 5 ---------------
        expShape = new Shape(6, 1);
        assertThrows(IllegalArgumentException.class, ()->A.reshape(expShape));

        // --------------- Sub-case 6 ---------------
        expShape = new Shape(12, 2);
        assertThrows(IllegalArgumentException.class, ()->A.reshape(expShape));
    }


    @Test
    void reshapeRowsColsTest() {
        // --------------- Sub-case 1 ---------------
        rows = 4;
        cols = 3;
        expShape = new Shape(rows, cols);
        B = A.reshape(rows, cols);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-case 2 ---------------
        rows = 1;
        cols = 12;
        expShape = new Shape(rows, cols);
        B = A.reshape(rows, cols);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-case 3 ---------------
        rows = 2;
        cols = 6;
        expShape = new Shape(rows, cols);
        B = A.reshape(rows, cols);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-case 4 ---------------
        rows = 6;
        cols = 2;
        expShape = new Shape(rows, cols);
        B = A.reshape(rows, cols);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-case 5 ---------------
        rows = 6;
        cols = 1;
        expShape = new Shape(rows, cols);
        assertThrows(IllegalArgumentException.class, ()->A.reshape(rows, cols));

        // --------------- Sub-case 6 ---------------
        rows = 12;
        cols = 2;
        expShape = new Shape(12, 2);
        assertThrows(IllegalArgumentException.class, ()->A.reshape(rows, cols));
    }


    @Test
    void flattenTest() {
        // --------------- Sub-case 1 ---------------
        expShape = new Shape(1, entries.length);
        B = A.flatten();
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-case 2 ---------------
        expShape = new Shape(1, entries.length);
        B = A.flatten(0);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-case 2 ---------------
        expShape = new Shape(entries.length, 1);
        B = A.flatten(1);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.entries, B.entries);

        // --------------- Sub-cases 2-4 ---------------
        assertThrows(IllegalArgumentException.class, ()->A.flatten(-1));
        assertThrows(IllegalArgumentException.class, ()->A.flatten(4));
        assertThrows(IllegalArgumentException.class, ()->A.flatten(2));
    }
}
