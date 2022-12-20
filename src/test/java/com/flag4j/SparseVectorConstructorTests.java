package com.flag4j;

import com.flag4j.core.VectorOrientation;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SparseVectorConstructorTests {
    int expSize;
    Shape expShape;
    double[] expEntries;
    int[] expEntriesI;
    int[] expIndices;
    VectorOrientation expOrientation;
    SparseVector a, b;


    @Test
    void sizeTest() {
        // ------------- Sub-case 1 -------------
        expSize = 114;
        expShape = new Shape(expSize);
        expEntries = new double[0];
        expIndices = new int[0];
        expOrientation = VectorOrientation.COL;
        a = new SparseVector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
        assertArrayEquals(expIndices, a.indices);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new SparseVector(expSize));
    }


    @Test
    void sizeEntriesIndicesTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientation.COL;
        a = new SparseVector(expSize, expEntries, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new SparseVector(expSize, expEntries, expIndices));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new SparseVector(expSize, expEntries, expIndices));


        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new SparseVector(expSize, expEntries, expIndices));
    }


    @Test
    void sizeEntriesIntIndicesTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientation.COL;
        a = new SparseVector(expSize, expEntriesI, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class, () -> new SparseVector(expSize, expEntriesI, expIndices));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new SparseVector(expSize, expEntriesI, expIndices));


        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new SparseVector(expSize, expEntriesI, expIndices));
    }


    @Test
    void sizeOrientationTest() {
        // ------------- Sub-case 1 -------------
        expSize = 114;
        expShape = new Shape(expSize);
        expEntries = new double[0];
        expIndices = new int[0];
        expOrientation = VectorOrientation.COL;
        a = new SparseVector(expSize, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
        assertArrayEquals(expIndices, a.indices);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new SparseVector(expSize, expOrientation));


        // ------------- Sub-case 3 -------------
        expSize = 114;
        expShape = new Shape(expSize);
        expEntries = new double[0];
        expIndices = new int[0];
        expOrientation = VectorOrientation.ROW;
        a = new SparseVector(expSize, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
        assertArrayEquals(expIndices, a.indices);
    }


    @Test
    void sizeOrientationEntriesIndicesTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientation.COL;
        a = new SparseVector(expSize, expEntries, expIndices, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expOrientation = VectorOrientation.COL;
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntries, expIndices, expOrientation));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntries, expIndices, expOrientation));

        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntries, expIndices, expOrientation));

        // ------------- Sub-case 5 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientation.ROW;
        a = new SparseVector(expSize, expEntries, expIndices, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 6 -------------
        expSize = -1;
        expOrientation = VectorOrientation.ROW;
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntries, expIndices, expOrientation));

        // ------------- Sub-case 7 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expOrientation = VectorOrientation.ROW;
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntries, expIndices, expOrientation));

        // ------------- Sub-case 8 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expOrientation = VectorOrientation.ROW;
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntries, expIndices, expOrientation));
    }


    @Test
    void sizeOrientationEntriesIntIndicesTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientation.COL;
        a = new SparseVector(expSize, expEntriesI, expIndices, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expOrientation = VectorOrientation.COL;
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntriesI, expIndices, expOrientation));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expOrientation = VectorOrientation.COL;
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntriesI, expIndices, expOrientation));

        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expOrientation = VectorOrientation.COL;
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntriesI, expIndices, expOrientation));

        // ------------- Sub-case 5 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientation.ROW;
        a = new SparseVector(expSize, expEntriesI, expIndices, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 6 -------------
        expSize = -1;
        expOrientation = VectorOrientation.ROW;
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntriesI, expIndices, expOrientation));

        // ------------- Sub-case 7 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expOrientation = VectorOrientation.ROW;
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntriesI, expIndices, expOrientation));

        // ------------- Sub-case 8 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expOrientation = VectorOrientation.ROW;
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseVector(expSize, expEntriesI, expIndices, expOrientation));
    }


    @Test
    void copyTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientation.COL;
        b = new SparseVector(expSize, expEntries, expIndices, expOrientation);
        a = new SparseVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 5 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientation.ROW;
        b = new SparseVector(expSize, expEntries, expIndices, expOrientation);
        a = new SparseVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
    }
}
