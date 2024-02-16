package com.flag4j.sparse_matrix;

import com.flag4j.core.Shape;
import com.flag4j.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class CooMatrixConstructorTests {
    double[] expNonZero;
    int[] expNonZeroI, expRowIndices, expColIndices;
    int size, rows, cols;
    Shape expShape;
    CooMatrix A, B;

    @Test
    void sizeTestCase() {
        // --------------- Sub-case 1 ---------------
        size = 5;
        expShape = new Shape(5, 5);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooMatrix(size);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        size = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size));
    }


    @Test
    void rowsColsTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooMatrix(rows, cols);

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
        A = new CooMatrix(rows, cols);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = -1;
        cols = 12;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols));

        // --------------- Sub-case 3 ---------------
        rows = 1;
        cols = -12;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols));

        // --------------- Sub-case 4 ---------------
        rows = -1;
        cols = -2;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols));
    }


    @Test
    void shapeTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooMatrix(expShape);

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
        A = new CooMatrix(expShape);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void sizeEntriesIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5, 6};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};
        A = new CooMatrix(size, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        size = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 3 ---------------
        size = 2;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5, 6, 5};
        expRowIndices = new int[]{0, 0, 1, 3, 4};
        expColIndices = new int[]{0, 4, 1, 3, 4};

        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};

        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5, 4};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1};

        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZero, expRowIndices, expColIndices));
    }


    @Test
    void rowColEntriesIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[]{-1, 0.234345, 133.1, 24.5, -933.1};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new CooMatrix(rows, cols, expNonZero, expRowIndices, expColIndices);

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
        A = new CooMatrix(rows, cols, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = -1;
        cols = 12;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 3 ---------------
        rows = 1;
        cols = -12;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        rows = -1;
        cols = -2;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));
    }


    @Test
    void shapeEntriesIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[]{-1, 0.234345, 133.1, 24.5, -933.1};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new CooMatrix(expShape, expNonZero, expRowIndices, expColIndices);

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
        A = new CooMatrix(expShape, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void sizeEntriesIntIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZeroI = new int[]{1, 2, 5, 6};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};
        A = new CooMatrix(size, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        size = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZeroI, expRowIndices, expColIndices));

        // --------------- Sub-case 3 ---------------
        size = 2;
        expShape = new Shape(size, size);
        expNonZeroI = new int[]{1, 2, 5, 6, 5};
        expRowIndices = new int[]{0, 0, 1, 3, 4};
        expColIndices = new int[]{0, 4, 1, 3, 4};

        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZeroI, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZeroI = new int[]{1, 2, 5};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};

        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZeroI, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZeroI = new int[]{1, 2, 5, 4};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1};

        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZeroI, expRowIndices, expColIndices));
    }


    @Test
    void rowColEntriesIntIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new CooMatrix(rows, cols, expNonZeroI, expRowIndices, expColIndices);

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
        A = new CooMatrix(rows, cols, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = -1;
        cols = 12;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols, expNonZeroI, expRowIndices, expColIndices));

        // --------------- Sub-case 3 ---------------
        rows = 1;
        cols = -12;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols, expNonZeroI, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        rows = -1;
        cols = -2;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols, expNonZeroI, expRowIndices, expColIndices));
    }

    @Test
    void shapeEntriesIntIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new CooMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);

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
        A = new CooMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void copyTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        B = new CooMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);
        A = new CooMatrix(B);

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
        B = new CooMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);
        A = new CooMatrix(B);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }
}
