package com.flag4j.operations;

import com.flag4j.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static com.flag4j.operations.RealDenseOperations.*;

class ReadDenseOperationsTests {

    double[] src1, src2;
    double[] expResult;
    double expResultD;
    Shape shape1;
    Shape shape2;
    double a;


    @Test
    void addTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4};
        expResult = new double[]{1+9.1233, 0.98332+22, 134.556+0.00009234,
                -9.13+11.234, -100.234-88.1, 0.0000000004+13.4};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        assertArrayEquals(expResult, add(src1, shape1, src2, shape2));

        // ---------- Sub-case 2 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4, 1, 1};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        assertThrows(IllegalArgumentException.class, () -> add(src1, shape1, src2, shape2));

        // ---------- Sub-case 3 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        assertThrows(IllegalArgumentException.class, () -> add(src1, shape1, src2, shape2));
    }


    @Test
    void addScalarTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        a = 11.93;
        expResult = new double[]{1+a, 0.98332+a, 134.556+a,
                -9.13+a, -100.234+a, 0.0000000004+a};

        assertArrayEquals(expResult, add(src1, a));
    }


    @Test
    void subTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4};
        expResult = new double[]{1-9.1233, 0.98332-22, 134.556-0.00009234,
                -9.13-11.234, -100.234+88.1, 0.0000000004-13.4};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertArrayEquals(expResult, sub(src1, shape1, src2, shape2));

        // ---------- Sub-case 2 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4, 1, 1};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(IllegalArgumentException.class, () -> sub(src1, shape1, src2, shape2));

        // ---------- Sub-case 3 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(IllegalArgumentException.class, () -> sub(src1, shape1, src2, shape2));
    }


    @Test
    void subScalarTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        a = 11.93;
        expResult = new double[]{1-a, 0.98332-a, 134.556-a,
                -9.13-a, -100.234-a, 0.0000000004-a};

        assertArrayEquals(expResult, sub(src1, a));
    }


    @Test
    void sumTest() {
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        expResultD = 1+0.98332+134.556+-9.13+-100.234+0.0000000004;
        assertEquals(expResultD, sum(src1));
    }

    @Test
    void prodTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.8000000004};
        expResultD = 1*0.98332*134.556*-9.13*-100.234*0.8000000004;
        assertEquals(expResultD, prod(src1));

        // ---------- Sub-case 2 -----------------
        src1 = new double[]{};
        expResultD = 0;
        assertEquals(expResultD, prod(src1));
    }


    @Test
    void scaleMultTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        a = 11.93;
        expResult = new double[]{1*a, 0.98332*a, 134.556*a,
                -9.13*a, -100.234*a, 0.0000000004*a};

        assertArrayEquals(expResult, scalMult(src1, a));
    }
}
