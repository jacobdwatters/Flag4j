package com.flag4j;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class SparseMatrixConstructorTests {
    double[] expNonZero;
    int[] expNonZeroI, expRowIndices, expColIndices;
    int size, rows, cols;
    Shape expShape;
    SparseMatrix A, B;

    @Test
    void sizeTest() {
        // --------------- Sub-case 1 ---------------
        size = 5;
        expShape = new Shape(5, 5);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new SparseMatrix(size);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        size = -1;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(size));
    }


    @Test
    void rowsColsTest() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new SparseMatrix(rows, cols);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new SparseMatrix(rows, cols);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = -1;
        cols = 12;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(rows, cols));

        // --------------- Sub-case 3 ---------------
        rows = 1;
        cols = -12;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(rows, cols));

        // --------------- Sub-case 4 ---------------
        rows = -1;
        cols = -2;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(rows, cols));
    }


    @Test
    void shapeTest() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new SparseMatrix(expShape);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new SparseMatrix(expShape);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void sizeEntriesIndicesTest() {
        // --------------- Sub-case 1 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5, 6};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};
        A = new SparseMatrix(size, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        size = -1;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 3 ---------------
        size = 2;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5, 6, 5};
        expRowIndices = new int[]{0, 0, 1, 3, 4};
        expColIndices = new int[]{0, 4, 1, 3, 4};

        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};

        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5, 4};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1};

        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(size, expNonZero, expRowIndices, expColIndices));
    }


    @Test
    void rowColEntriesIndicesTest() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[]{-1, 0.234345, 133.1, 24.5, -933.1};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new SparseMatrix(rows, cols, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new double[]{-1, 0.234345, 133.1, 24.5, -933.1};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        A = new SparseMatrix(rows, cols, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = -1;
        cols = 12;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 3 ---------------
        rows = 1;
        cols = -12;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        rows = -1;
        cols = -2;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));
    }


    @Test
    void shapeEntriesIndicesTest() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[]{-1, 0.234345, 133.1, 24.5, -933.1};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new SparseMatrix(expShape, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new double[]{-1, 0.234345, 133.1, 24.5, -933.1};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        A = new SparseMatrix(expShape, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void sizeEntriesIntIndicesTest() {
        // --------------- Sub-case 1 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZeroI = new int[]{1, 2, 5, 6};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};
        A = new SparseMatrix(size, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        size = -1;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(size, expNonZeroI, expRowIndices, expColIndices));

        // --------------- Sub-case 3 ---------------
        size = 2;
        expShape = new Shape(size, size);
        expNonZeroI = new int[]{1, 2, 5, 6, 5};
        expRowIndices = new int[]{0, 0, 1, 3, 4};
        expColIndices = new int[]{0, 4, 1, 3, 4};

        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(size, expNonZeroI, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZeroI = new int[]{1, 2, 5};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};

        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(size, expNonZeroI, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZeroI = new int[]{1, 2, 5, 4};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1};

        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(size, expNonZeroI, expRowIndices, expColIndices));
    }


    @Test
    void rowColEntriesIntIndicesTest() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new SparseMatrix(rows, cols, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        A = new SparseMatrix(rows, cols, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = -1;
        cols = 12;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(rows, cols, expNonZeroI, expRowIndices, expColIndices));

        // --------------- Sub-case 3 ---------------
        rows = 1;
        cols = -12;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(rows, cols, expNonZeroI, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        rows = -1;
        cols = -2;
        assertThrows(IllegalArgumentException.class, () -> new SparseMatrix(rows, cols, expNonZeroI, expRowIndices, expColIndices));
    }

    @Test
    void shapeEntriesIntIndicesTest() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new SparseMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        A = new SparseMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void copyTest() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        B = new SparseMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);
        A = new SparseMatrix(B);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        B = new SparseMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);
        A = new SparseMatrix(B);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }
}
