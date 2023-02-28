package com.flag4j.matrix.vector;

import com.flag4j.Matrix;
import com.flag4j.Vector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class VectorExtendTests {

    int axis, n;
    double[] aEntries;
    Vector a;
    double[][] expEntries;
    Matrix exp;

    @Test
    void extendTest() {
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
