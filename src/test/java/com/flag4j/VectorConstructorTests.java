package com.flag4j;


import com.flag4j.core.VectorOrientations;

import java.util.Arrays;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class VectorConstructorTests {

    int expSize;
    double fillValue;
    Shape expShape;
    VectorOrientations expOrientation;
    double[] expEntries;
    int[] entriesI;
    Vector a, b;

    @Test
    void sizeTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.COL;
        expEntries = new double[expSize];

        a = new Vector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.COL;
        expEntries = new double[expSize];

        a = new Vector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expSize = -1;
        assertThrows(NegativeArraySizeException.class, () -> new Vector(expSize));
    }

    @Test
    void sizeFillTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        fillValue = -10.23423;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.COL;
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new Vector(expSize, fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        fillValue = -10.23423;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.COL;
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new Vector(expSize,fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expSize = -1;
        fillValue = -10.23423;
        assertThrows(NegativeArraySizeException.class, () -> new Vector(expSize, fillValue));
    }


    @Test
    void sizeOrientationTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.COL;
        expEntries = new double[expSize];

        a = new Vector(expSize, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.COL;
        expEntries = new double[expSize];

        a = new Vector(expSize, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expSize = 5;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.ROW;
        expEntries = new double[expSize];

        a = new Vector(expSize, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 4 ------------
        expSize = 0;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.ROW;
        expEntries = new double[expSize];

        a = new Vector(expSize, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 5 ------------
        expSize = -1;
        assertThrows(NegativeArraySizeException.class, () -> new Vector(expSize, expOrientation));
    }


    @Test
    void sizeFillOrientationTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        fillValue = -0.12334;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.COL;
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new Vector(expSize, fillValue, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        fillValue = -0.12334;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.COL;
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new Vector(expSize, fillValue, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expSize = 5;
        fillValue = -0.12334;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.ROW;
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new Vector(expSize, fillValue, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 4 ------------
        expSize = 0;
        fillValue = -0.12334;
        expShape = new Shape(new int[]{expSize});
        expOrientation = VectorOrientations.ROW;
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new Vector(expSize, fillValue, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 5 ------------
        expSize = -1;
        assertThrows(NegativeArraySizeException.class,
                () -> new Vector(expSize, fillValue, expOrientation));
    }





    @Test
    void entriesTest() {
        // ----------- Sub-case 1 ------------
        expEntries = new double[]{1.0433, 2, -3, 4, 5, 6, 7, 100, -0.1231};
        expSize = expEntries.length;
        expOrientation = VectorOrientations.COL;
        expShape = new Shape(expSize);

        a = new Vector(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expEntries = new double[]{-0.234974};
        expSize = expEntries.length;
        expOrientation = VectorOrientations.COL;
        expShape = new Shape(expSize);

        a = new Vector(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
    }


    @Test
    void entriesOrientationTest() {
        // ----------- Sub-case 1 ------------
        expEntries = new double[]{1.0433, 2, -3, 4, 5, 6, 7, 100, -0.1231};
        expSize = expEntries.length;
        expOrientation = VectorOrientations.COL;
        expShape = new Shape(expSize);

        a = new Vector(expEntries, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expEntries = new double[]{-0.234974};
        expSize = expEntries.length;
        expOrientation = VectorOrientations.COL;
        expShape = new Shape(expSize);

        a = new Vector(expEntries, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expEntries = new double[]{1.0433, 2, -3, 4, 5, 6, 7, 100, -0.1231};
        expSize = expEntries.length;
        expOrientation = VectorOrientations.ROW;
        expShape = new Shape(expSize);

        a = new Vector(expEntries, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 4 ------------
        expEntries = new double[]{-0.234974};
        expSize = expEntries.length;
        expOrientation = VectorOrientations.ROW;
        expShape = new Shape(expSize);

        a = new Vector(expEntries, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
    }


    @Test
    void entriesITest() {
        // ----------- Sub-case 1 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expOrientation = VectorOrientations.COL;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        a = new Vector(entriesI);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expOrientation = VectorOrientations.COL;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        a = new Vector(entriesI);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
    }


    @Test
    void entriesIOrientationTest() {
        // ----------- Sub-case 1 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expOrientation = VectorOrientations.COL;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        a = new Vector(entriesI, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expOrientation = VectorOrientations.COL;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        a = new Vector(entriesI, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expOrientation = VectorOrientations.ROW;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        a = new Vector(entriesI, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 4 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expOrientation = VectorOrientations.ROW;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        a = new Vector(entriesI, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
    }


    @Test
    void copyTest() {
        // ----------- Sub-case 1 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expOrientation = VectorOrientations.COL;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        b = new Vector(entriesI, expOrientation);
        a = new Vector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expOrientation = VectorOrientations.COL;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        b = new Vector(entriesI, expOrientation);
        a = new Vector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expOrientation = VectorOrientations.ROW;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        b = new Vector(entriesI, expOrientation);
        a = new Vector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 4 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expOrientation = VectorOrientations.ROW;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        b = new Vector(entriesI, expOrientation);
        a = new Vector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
    }
}
