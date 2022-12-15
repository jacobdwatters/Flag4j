package com.flag4j.operations;

import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static com.flag4j.operations.ComplexDenseOperations.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexDenseOperationsTests {

    CNumber[] src1, src2;
    CNumber[] expResult;
    CNumber expResultC;
    Shape shape1, shape2;
    double a;
    CNumber aC;


    @Test
    void addTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(-9234.23), new CNumber(109.2234, 1.435),
                new CNumber(0, -1943.134), new CNumber(-9234.1, -3245)};
        expResult = new CNumber[]{new CNumber(9-9234.23, -1), new CNumber(-0.99+109.2234, 13.445+1.435),
                new CNumber(0.9133, -1943.134), new CNumber(-9234.1, 10.3-3245)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertArrayEquals(expResult, add(src1, shape1, src2, shape2));

        // ---------- Sub-case 2 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(9-9234.23, -1), new CNumber(-0.99+109.2234, 13.445+1.435),
                new CNumber(0.9133, -1943.134), new CNumber(-9234.1, 10.3-3245), new CNumber(0, 1)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(IllegalArgumentException.class, () -> add(src1, shape1, src2, shape2));

        // ---------- Sub-case 3 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(9-9234.23, -1), new CNumber(-0.99+109.2234, 13.445+1.435)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(IllegalArgumentException.class, () -> add(src1, shape1, src2, shape2));
    }


    @Test
    void addDoubleTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        a = 933.1334;
        expResult = new CNumber[]{new CNumber(9+a, -1), new CNumber(-0.99+a, 13.445),
                new CNumber(0.9133+a), new CNumber(0+a, 10.3)};
        assertArrayEquals(expResult, add(src1, a));
    }


    @Test
    void addCNumberTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        aC = new CNumber(10.34, -1.334);
        expResult = new CNumber[]{new CNumber(9, -1).add(aC), new CNumber(-0.99, 13.445).add(aC),
                new CNumber(0.9133).add(aC), new CNumber(0, 10.3).add(aC)};
        assertArrayEquals(expResult, add(src1, aC));
    }


    @Test
    void subTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(-9234.23), new CNumber(109.2234, 1.435),
                new CNumber(0, -1943.134), new CNumber(-9234.1, -3245)};

        expResult = new CNumber[]{new CNumber(9+9234.23, -1), new CNumber(-0.99-109.2234, 13.445-1.435),
                new CNumber(0.9133, 1943.134), new CNumber(9234.1, 10.3+3245)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertArrayEquals(expResult, sub(src1, shape1, src2, shape2));

        // ---------- Sub-case 2 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(9-9234.23, -1), new CNumber(-0.99+109.2234, 13.445+1.435),
                new CNumber(0.9133, -1943.134), new CNumber(-9234.1, 10.3-3245), new CNumber(0, 1)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(IllegalArgumentException.class, () -> sub(src1, shape1, src2, shape2));

        // ---------- Sub-case 3 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(9-9234.23, -1), new CNumber(-0.99+109.2234, 13.445+1.435)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(IllegalArgumentException.class, () -> sub(src1, shape1, src2, shape2));
    }


    @Test
    void subDoubleTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        a = 933.1334;
        expResult = new CNumber[]{new CNumber(9-a, -1), new CNumber(-0.99-a, 13.445),
                new CNumber(0.9133-a), new CNumber(0-a, 10.3)};
        assertArrayEquals(expResult, sub(src1, a));
    }


    @Test
    void subCNumberTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        aC = new CNumber(10.34, -1.334);
        expResult = new CNumber[]{new CNumber(9, -1).sub(aC), new CNumber(-0.99, 13.445).sub(aC),
                new CNumber(0.9133).sub(aC), new CNumber(0, 10.3).sub(aC)};
        assertArrayEquals(expResult, sub(src1, aC));
    }


    @Test
    void sumTest() {
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        expResultC = new CNumber(9-0.99+0.9133, -1+13.445+10.3);
        assertEquals(expResultC, sum(src1));
    }


    @Test
    void prodTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};

        expResultC = new CNumber(-1147.60574505, 42.660699650000005);
        assertEquals(expResultC, prod(src1));

        // ---------- Sub-case 2 -----------------
        src1 = new CNumber[]{};
        expResultC = new CNumber();
        assertEquals(expResultC, prod(src1));
    }


    @Test
    void scalMultDoubleTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        a = 933.1334;
        expResult = new CNumber[]{new CNumber(9*a, -1*a), new CNumber(-0.99*a, 13.445*a),
                new CNumber(0.9133*a), new CNumber(0*a, 10.3*a)};
        assertArrayEquals(expResult, scalMult(src1, a));
    }


    @Test
    void scalMultCNumberTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        aC = new CNumber(10.34, -1.334);
        expResult = new CNumber[]{new CNumber(9, -1).mult(aC), new CNumber(-0.99, 13.445).mult(aC),
                new CNumber(0.9133).mult(aC), new CNumber(0, 10.3).mult(aC)};
        assertArrayEquals(expResult, scalMult(src1, aC));
    }
}
