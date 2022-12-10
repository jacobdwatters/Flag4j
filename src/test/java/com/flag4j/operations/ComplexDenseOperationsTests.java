package com.flag4j.operations;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static com.flag4j.operations.ComplexDenseOperations.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexDenseOperationsTests {

    CNumber[] src1, src2;
    CNumber[] expResult;
    CNumber expResultC;


    @Test
    void addTest() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(-9234.23), new CNumber(109.2234, 1.435),
                new CNumber(0, -1943.134), new CNumber(-9234.1, -3245)};
        expResult = new CNumber[]{new CNumber(9-9234.23, -1), new CNumber(-0.99+109.2234, 13.445+1.435),
                new CNumber(0.9133, -1943.134), new CNumber(-9234.1, 10.3-3245)};
        assertArrayEquals(expResult, add(src1, src2));

        // ---------- Sub-case 2 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(9-9234.23, -1), new CNumber(-0.99+109.2234, 13.445+1.435),
                new CNumber(0.9133, -1943.134), new CNumber(-9234.1, 10.3-3245), new CNumber(0, 1)};
        assertThrows(IllegalArgumentException.class, () -> add(src1, src2));

        // ---------- Sub-case 3 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(9-9234.23, -1), new CNumber(-0.99+109.2234, 13.445+1.435)};
        assertThrows(IllegalArgumentException.class, () -> add(src1, src2));
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
        assertArrayEquals(expResult, sub(src1, src2));

        // ---------- Sub-case 2 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(9-9234.23, -1), new CNumber(-0.99+109.2234, 13.445+1.435),
                new CNumber(0.9133, -1943.134), new CNumber(-9234.1, 10.3-3245), new CNumber(0, 1)};
        assertThrows(IllegalArgumentException.class, () -> sub(src1, src2));

        // ---------- Sub-case 3 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new CNumber[]{new CNumber(9-9234.23, -1), new CNumber(-0.99+109.2234, 13.445+1.435)};
        assertThrows(IllegalArgumentException.class, () -> sub(src1, src2));
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
}
