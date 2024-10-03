package org.flag4j.sparse_complex_matrix;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class CooCMatrixConstructorTests {
    Complex128[] expNonZero;
    double[] expNonZeroD;
    int[] expNonZeroI, expRowIndices, expColIndices;
    int size, rows, cols;
    Shape expShape;
    CooCMatrix A, B;

    @Test
    void sizeTestCase() {
        // --------------- Sub-case 1 ---------------
        size = 5;
        expShape = new Shape(5, 5);
        expNonZero = new Complex128[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooCMatrix(size);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        size = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooCMatrix(size));
    }


    @Test
    void rowsColsTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new Complex128[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooCMatrix(rows, cols);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new Complex128[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooCMatrix(rows, cols);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = -1;
        cols = 12;
        assertThrows(IllegalArgumentException.class, () -> new CooCMatrix(rows, cols));

        // --------------- Sub-case 3 ---------------
        rows = 1;
        cols = -12;
        assertThrows(IllegalArgumentException.class, () -> new CooCMatrix(rows, cols));

        // --------------- Sub-case 4 ---------------
        rows = -1;
        cols = -2;
        assertThrows(IllegalArgumentException.class, () -> new CooCMatrix(rows, cols));
    }


    @Test
    void shapeTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new Complex128[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooCMatrix(expShape);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new Complex128[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooCMatrix(expShape);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void sizeEntriesIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new Complex128[]{new Complex128(0.1233, -90932), new Complex128(13, 11.2),
                new Complex128(-99.134, 3), new Complex128(0, 100)};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};
        A = new CooCMatrix(size, expNonZero, expRowIndices, expColIndices);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        size = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooCMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 3 ---------------
        size = 2;
        expShape = new Shape(size, size);
        expNonZero = new Complex128[]{new Complex128(0.1233, -90932), new Complex128(13, 11.2),
                new Complex128(-99.134, 3), new Complex128(0, 100),
                new Complex128(101)};
        expRowIndices = new int[]{0, 0, 1, 3, 4};
        expColIndices = new int[]{0, 4, 1, 3, 4};

        assertThrows(IndexOutOfBoundsException.class, () -> new CooCMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new Complex128[]{new Complex128(0.1233, -90932), new Complex128(13, 11.2),
                new Complex128(-99.134, 3)};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};

        assertThrows(IllegalArgumentException.class, () -> new CooCMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new Complex128[]{new Complex128(0.1233, -90932), new Complex128(13, 11.2),
                new Complex128(-99.134, 3), new Complex128(0, 100)};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1};

        assertThrows(IllegalArgumentException.class, () -> new CooCMatrix(size, expNonZero, expRowIndices, expColIndices));
    }


    @Test
    void rowColEntriesIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 11;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new Complex128[]{new Complex128(0.1233, -90932), new Complex128(13, 11.2),
                new Complex128(-99.134, 3), new Complex128(0, 100),
                new Complex128(101)};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new CooCMatrix(rows, cols, expNonZero, expRowIndices, expColIndices);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new Complex128[]{new Complex128(0.1233, -90932), new Complex128(13, 11.2),
                new Complex128(-99.134, 3), new Complex128(0, 100),
                new Complex128(101)};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        A = new CooCMatrix(rows, cols, expNonZero, expRowIndices, expColIndices);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = -1;
        cols = 12;
        assertThrows(IllegalArgumentException.class, () -> new CooCMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 3 ---------------
        rows = 1;
        cols = -12;
        assertThrows(IllegalArgumentException.class, () -> new CooCMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));

        // --------------- Sub-case 4 ---------------
        rows = -1;
        cols = -2;
        assertThrows(IllegalArgumentException.class, () -> new CooCMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));
    }


    @Test
    void shapeEntriesIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 11;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new Complex128[]{new Complex128(0.1233, -90932), new Complex128(13, 11.2),
                new Complex128(-99.134, 3), new Complex128(0, 100),
                new Complex128(101)};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new CooCMatrix(expShape, expNonZero, expRowIndices, expColIndices);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new Complex128[]{new Complex128(0.1233, -90932), new Complex128(13, 11.2),
                new Complex128(-99.134, 3), new Complex128(0, 100),
                new Complex128(101)};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        A = new CooCMatrix(expShape, expNonZero, expRowIndices, expColIndices);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void copyTestCase() {
        // --------------- Sub-case 1 ---------------
        rows = 11;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new Complex128[]{new Complex128(0.1233, -90932), new Complex128(13, 11.2),
                new Complex128(-99.134, 3), new Complex128(0, 100),
                new Complex128(101)};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        B = new CooCMatrix(rows, cols, expNonZero, expRowIndices, expColIndices);
        A = new CooCMatrix(B);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- Sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new Complex128[]{new Complex128(0.1233, -90932), new Complex128(13, 11.2),
                new Complex128(-99.134, 3), new Complex128(0, 100),
                new Complex128(101)};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        B = new CooCMatrix(rows, cols, expNonZero, expRowIndices, expColIndices);
        A = new CooCMatrix(B);

        Assertions.assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }
}
