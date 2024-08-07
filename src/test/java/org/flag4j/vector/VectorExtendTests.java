package org.flag4j.vector;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorExtendTests {

    int axis, n;
    double[] aEntries;
    Vector a;
    double[][] expEntries;
    Matrix exp;

    @Test
    void extendTestCase() {
        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[]{1.0, 13.5, 790091.1, -9.234};
        a = new Vector(aEntries);
        axis = 0;
        n = 5;
        expEntries = new double[][]{{1.0, 13.5, 790091.1, -9.234}, {1.0, 13.5, 790091.1, -9.234},
                {1.0, 13.5, 790091.1, -9.234}, {1.0, 13.5, 790091.1, -9.234}, {1.0, 13.5, 790091.1, -9.234}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.extend(n, axis));

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[]{1.0, 13.5, 790091.1, -9.234};
        a = new Vector(aEntries);
        axis = 1;
        n = 3;
        expEntries = new double[][]{
                {1.0, 1.0, 1.0},
                {13.5, 13.5, 13.5},
                {790091.1, 790091.1, 790091.1},
                {-9.234, -9.234, -9.234}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.extend(n, axis));

        // --------------------- Sub-case 3 ---------------------
        aEntries = new double[]{1.0, 13.5, 790091.1, -9.234};
        a = new Vector(aEntries);
        axis = 2;
        n = 5;

        assertThrows(IllegalArgumentException.class, ()->a.extend(n, axis));

        // --------------------- Sub-case 3 ---------------------
        aEntries = new double[]{1.0, 13.5, 790091.1, -9.234};
        a = new Vector(aEntries);
        axis = 1;
        n = -1;

        assertThrows(IllegalArgumentException.class, ()->a.extend(n, axis));
    }
}
