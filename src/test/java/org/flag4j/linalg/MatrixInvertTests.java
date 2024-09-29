package org.flag4j.linalg;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.io.PrintOptions;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.SingularMatrixException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixInvertTests {

    static Matrix A;
    static Matrix exp;
    static double[][] entries;
    static double[][] expEntries;

    @Test
    void invTriUTest() {
        // --------------------- Sub-case 1 ---------------------
        entries = new double[][]{
                {0.61158, 0.87596},
                {0.0, 0.45014}};
        A = new Matrix(entries);
        expEntries = new double[][]{
                {1.6351090617744204, -3.1818770465897748},
                {0.0, 2.2215310792197984}};
        exp = new Matrix(expEntries);

        assertEquals(exp, Invert.invTriU(A));

        // --------------------- Sub-case 2 ---------------------
        entries = new double[][]{
                {0.76978, 0.27277, 0.90265},
                {0.0, 0.1561, 0.83679},
                {0.0, 0.0, 0.00644}};
        A = new Matrix(entries);
        expEntries = new double[][]{
                {1.299072462261945, -2.270006377522042, 112.87436002887036},
                {0.0, 6.406149903907752, -832.3916425607092},
                {0.0, 0.0, 155.27950310559004}};
        exp = new Matrix(expEntries);

        assertEquals(exp, Invert.invTriU(A));

        // --------------------- Sub-case 3 ---------------------
        entries = new double[][]{
                {0.31673, 0.69142, 0.08349, 0.07415},
                {0.0, 0.24664, 0.62689, 0.797},
                {0.0, 0.0, 0.50857, 0.01702},
                {0.0, 0.0, 0.0, 0.11526}};
        A = new Matrix(entries);
        expEntries = new double[][]{
                {3.157263284185268, -8.850936506452229, 10.391811697373441, 57.6367923653548},
                {0.0, 4.05449237755433, -4.997779512289427, -27.29800639954568},
                {0.0, 0.0, 1.9662976581394893, -0.2903555972716823},
                {0.0, 0.0, 0.0, 8.676036786395974}};
        exp = new Matrix(expEntries);

        assertEquals(exp, Invert.invTriU(A));

        // --------------------- Sub-case 4 ---------------------
        entries = new double[][]{
                {0.21671, 0.00066, 0.09873, 0.57298, 0.22919, 0.6064, 0.53715, 0.36795, 0.22719, 0.47063},
                {0.0, 0.77367, 0.70667, 0.43124, 0.64431, 0.77275, 0.30505, 0.83899, 0.3178, 0.50588},
                {0.0, 0.0, 0.36696, 0.55825, 0.82823, 0.26704, 0.31807, 0.80966, 0.39851, 0.23385},
                {0.0, 0.0, 0.0, 0.63701, 0.01634, 0.58154, 0.09358, 0.35212, 0.06829, 0.38112},
                {0.0, 0.0, 0.0, 0.0, 0.22453, 0.3002, 0.44086, 0.68154, 0.74422, 0.2514},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.71469, 0.49553, 0.78324, 0.88894, 0.37997},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47257, 0.7119, 0.92818, 0.42631},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29617, 0.65823, 0.28397},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38876, 0.3932},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22844}};
        A = new Matrix(entries);
        expEntries = new double[][]{
                {4.614461723040007, -0.0039364906707076725, -1.2339328702132952, -3.0665981398000906, 0.07587526255554107, -0.9865627105790509, -2.8410329779241246, 10.560903663350672, -9.877411328020631, 7.614045866631429},
                {0.0, 1.2925407473470603, -2.489099002419193, 1.3063283876462979, 5.377473531360279, -3.7892255797235195, -0.46102392303739986, 0.3444759703335606, 0.15302971143503913, -1.9402048857914176},
                {0.0, 0.0, 2.725092653150207, -2.388161839878657, -9.878327723043597, 5.07433555348961, 2.53335596408712, -1.387453772756238, 1.2342665604944696, -1.5018548065037962},
                {0.0, 0.0, 0.0, 1.5698340685389556, -0.11424348051452606, -1.229379746977967, 1.0848231049490034, -0.9599035887977831, 1.7892514057757696, -4.359427837733901},
                {0.0, 0.0, 0.0, 0.0, 4.453747828797933, -1.8707622860333004, -2.193242127527721, -0.029653144272670434, 1.0383508706701416, 0.5528954389871591},
                {0.0, 0.0, 0.0, 0.0, 0.0, 1.3992080482446934, -1.4671891236148993, -0.1736292487616131, 0.5975192374796691, -0.4019381597331419},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1160886217914805, -5.086414862590252, 3.5598305330496074, -3.7534839669991142},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.376439207212074, -5.716826780952783, 5.642833341790496},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5722810988784857, -4.427512379964194},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.377517072316582}};
        exp = new Matrix(expEntries);

        assertEquals(exp, Invert.invTriU(A));

        // --------------------- Sub-case 5 ---------------------
        entries = new double[][]{
                {0.61158, 0.87596},
                {0.0, 0}};
        A = new Matrix(entries);

        assertThrows(SingularMatrixException.class, ()-> Invert.invTriU(A));

        // --------------------- Sub-case 6 ---------------------
        entries = new double[][]{
                {-0, 0.87596},
                {0.0, 14.5}};
        A = new Matrix(entries);

        assertThrows(SingularMatrixException.class, ()-> Invert.invTriU(A));

        // --------------------- Sub-case 7 ---------------------
        entries = new double[][]{
                {-1.35E-98, 0.87596, 14.45},
                {0.0, 14.5, 109.1},
                {0.0, 0.0, 14587.1}};
        A = new Matrix(entries);

        assertThrows(SingularMatrixException.class, ()-> Invert.invTriU(A));

        // --------------------- Sub-case 8 ---------------------
        entries = new double[][]{
                {-1.35E-98, 0.87596, 14.45},
                {0.0, 14.5, 109.1}};
        A = new Matrix(entries).T();

        assertThrows(LinearAlgebraException.class, ()-> Invert.invTriL(A));

        // --------------------- Sub-case 9 ---------------------
        entries = new double[][]{
                {-1.35E-98, 0.87596},
                {0.0, 14.5},
                {0.0, 0.0}};
        A = new Matrix(entries).T();

        assertThrows(LinearAlgebraException.class, ()-> Invert.invTriL(A));
    }


    @Test
    void invTriLTest() {
        PrintOptions.setPrecision(100);

        // --------------------- Sub-case 1 ---------------------
        entries = new double[][]{
                {0.61158, 0.87596},
                {0.0, 0.45014}};
        A = new Matrix(entries).T();
        expEntries = new double[][]{
                {1.6351090617744204, -3.1818770465897748},
                {0.0, 2.2215310792197984}};
        exp = new Matrix(expEntries).T();

        assertEquals(exp, Invert.invTriL(A));

        // --------------------- Sub-case 2 ---------------------
        entries = new double[][]{
                {0.76978, 0.27277, 0.90265},
                {0.0, 0.1561, 0.83679},
                {0.0, 0.0, 0.00644}};
        A = new Matrix(entries).T();
        expEntries = new double[][]{
                {1.299072462261945, -2.270006377522042, 112.87436002887034},
                {0.0, 6.406149903907752, -832.3916425607092},
                {0.0, 0.0, 155.27950310559004}};
        exp = new Matrix(expEntries).T();

        assertEquals(exp, Invert.invTriL(A));

        // --------------------- Sub-case 3 ---------------------
        entries = new double[][]{
                {0.31673, 0.69142, 0.08349, 0.07415},
                {0.0, 0.24664, 0.62689, 0.797},
                {0.0, 0.0, 0.50857, 0.01702},
                {0.0, 0.0, 0.0, 0.11526}};
        A = new Matrix(entries).T();
        expEntries = new double[][]{
                {3.157263284185268, -8.85093650645223, 10.391811697373441, 57.63679236535481},
                {0.0, 4.05449237755433, -4.997779512289427, -27.298006399545677},
                {0.0, 0.0, 1.9662976581394893, -0.29035559727168236},
                {0.0, 0.0, 0.0, 8.676036786395974}};
        exp = new Matrix(expEntries).T();

        assertEquals(exp, Invert.invTriL(A));

        // --------------------- Sub-case 4 ---------------------
        entries = new double[][]{
                {0.21671, 0.00066, 0.09873, 0.57298, 0.22919, 0.6064, 0.53715, 0.36795, 0.22719, 0.47063},
                {0.0, 0.77367, 0.70667, 0.43124, 0.64431, 0.77275, 0.30505, 0.83899, 0.3178, 0.50588},
                {0.0, 0.0, 0.36696, 0.55825, 0.82823, 0.26704, 0.31807, 0.80966, 0.39851, 0.23385},
                {0.0, 0.0, 0.0, 0.63701, 0.01634, 0.58154, 0.09358, 0.35212, 0.06829, 0.38112},
                {0.0, 0.0, 0.0, 0.0, 0.22453, 0.3002, 0.44086, 0.68154, 0.74422, 0.2514},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.71469, 0.49553, 0.78324, 0.88894, 0.37997},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47257, 0.7119, 0.92818, 0.42631},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29617, 0.65823, 0.28397},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38876, 0.3932},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22844}};
        A = new Matrix(entries).T();
        expEntries = new double[][]{
                {4.614461723040007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {-0.0039364906707076725, 1.2925407473470603, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {-1.2339328702132955, -2.489099002419193, 2.725092653150207, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {-3.0665981398000897, 1.3063283876462977, -2.3881618398786566, 1.5698340685389556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.07587526255554022, 5.377473531360278, -9.878327723043595, -0.11424348051452604, 4.453747828797933, 0.0, 0.0, 0.0, 0.0, 0.0},
                {-0.9865627105790509, -3.789225579723521, 5.074335553489611, -1.229379746977967, -1.8707622860333004, 1.3992080482446934, 0.0, 0.0, 0.0, 0.0},
                {-2.8410329779241246, -0.4610239230374004, 2.5333559640871197, 1.0848231049490031, -2.193242127527722, -1.4671891236148993, 2.1160886217914805, 0.0, 0.0, 0.0},
                {10.56090366335067, 0.34447597033356486, -1.3874537727562395, -0.9599035887977826, -0.029653144272668908, -0.1736292487616129, -5.086414862590252, 3.376439207212074, 0.0, 0.0},
                {-9.877411328020631, 0.15302971143503333, 1.2342665604944718, 1.7892514057757691, 1.0383508706701405, 0.5975192374796684, 3.559830533049607, -5.716826780952782, 2.5722810988784857, 0.0},
                {7.614045866631427, -1.940204885791414, -1.5018548065037949, -4.359427837733902, 0.552895438987158, -0.40193815973314095, -3.753483966999114, 5.642833341790497, -4.4275123799641944, 4.377517072316582}};
        exp = new Matrix(expEntries);

        assertEquals(exp, Invert.invTriL(A));

        // --------------------- Sub-case 5 ---------------------
        entries = new double[][]{
                {0.61158, 0.87596},
                {0.0, 0}};
        A = new Matrix(entries).T();

        assertThrows(SingularMatrixException.class, ()-> Invert.invTriL(A));

        // --------------------- Sub-case 6 ---------------------
        entries = new double[][]{
                {-0, 0.87596},
                {0.0, 14.5}};
        A = new Matrix(entries).T();

        assertThrows(SingularMatrixException.class, ()-> Invert.invTriL(A));

        // --------------------- Sub-case 7 ---------------------
        entries = new double[][]{
                {-1.35E-98, 0.87596, 14.45},
                {0.0, 14.5, 109.1},
                {0.0, 0.0, 14587.1}};
        A = new Matrix(entries).T();

        assertThrows(SingularMatrixException.class, ()-> Invert.invTriL(A));

        // --------------------- Sub-case 8 ---------------------
        entries = new double[][]{
                {-1.35E-98, 0.87596, 14.45},
                {0.0, 14.5, 109.1}};
        A = new Matrix(entries).T();

        assertThrows(LinearAlgebraException.class, ()-> Invert.invTriL(A));

        // --------------------- Sub-case 9 ---------------------
        entries = new double[][]{
                {-1.35E-98, 0.87596},
                {0.0, 14.5},
                {0.0, 0.0}};
        A = new Matrix(entries).T();

        assertThrows(LinearAlgebraException.class, ()-> Invert.invTriL(A));
    }


    @Test
    void invDiagTests() {
        // --------------------- Sub-case 1 ---------------------
        entries = new double[][]{
                {0.61158, 0.0},
                {0.0, 0.45014}};
        A = new Matrix(entries);
        expEntries = new double[][]{
                {1.0/0.61158, 0.0},
                {0.0, 1.0/0.45014}};
        exp = new Matrix(expEntries);

        assertEquals(exp, Invert.invDiag(A));

        // --------------------- Sub-case 2 ---------------------
        entries = new double[][]{
                {0.31673, 0.0, 0.0, 0.0},
                {0.0, 0.24664, 0.0, 0.0},
                {0.0, 0.0, 0.50857, 0.0},
                {0.0, 0.0, 0.0, 0.11526}};
        A = new Matrix(entries);
        expEntries = new double[][]{
                {1.0/0.31673, 0.0, 0.0, 0.0},
                {0.0, 1.0/0.24664, 0.0, 0.0},
                {0.0, 0.0, 1.0/0.50857, 0.0},
                {0.0, 0.0, 0.0, 1.0/0.11526}};
        exp = new Matrix(expEntries);

        assertEquals(exp, Invert.invDiag(A));

        // --------------------- Sub-case 3 ---------------------
        entries = new double[][]{
                {0.31673, 0.0, 0.0, 0.0},
                {0.0, 0.24664, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.11526}};
        A = new Matrix(entries);

        assertThrows(SingularMatrixException.class, ()-> Invert.invDiag(A));

        // --------------------- Sub-case 4 ---------------------
        entries = new double[][]{
                {0.31673, 0.0, 0.0, 0.0},
                {0.0, 0.24664, 0.0, 0.0},
                {0.0, 0.0, -40.0, 0.0},
                {0.0, 0.0, 0.0, 1.1526E-71}};
        A = new Matrix(entries);

        assertThrows(SingularMatrixException.class, ()-> Invert.invDiag(A));

        // --------------------- Sub-case 5 ---------------------
        entries = new double[][]{
                {0.31673, 0.0, 0.0, 0.0},
                {0.0, 0.24664, 0.0, 0.0},
                {0.0, 0.0, -40.0, 0.0}};
        A = new Matrix(entries);

        assertThrows(LinearAlgebraException.class, ()-> Invert.invDiag(A));

        // --------------------- Sub-case 6 ---------------------
        entries = new double[][]{
                {0.31673},
                {6.0},
                {-24.15}};
        A = new Matrix(entries);

        assertThrows(LinearAlgebraException.class, ()-> Invert.invDiag(A));
    }


    @Test
    void symmPosDefInvertTest() {
        // --------------------- Sub-case 1 ---------------------
        entries = new double[][]{
                {2, -1, 0},
                {-1, 2, -1},
                {0, -1, 2}};
        A = new Matrix(entries);
        expEntries = new double[][]{
                {0.7499999999999999, 0.5000000000000001, 0.25000000000000006},
                {0.5, 1.0000000000000004, 0.5000000000000002},
                {0.25, 0.5000000000000002, 0.7500000000000001}};
        exp = new Matrix(expEntries);

        assertEquals(exp, Invert.invSymPosDef(A));

        // --------------------- Sub-case 2 ---------------------
        entries = new double[][]{
                {4, 12, -16},
                {12, 37, -43},
                {-16, -43, 98}};
        A = new Matrix(entries);
        expEntries = new double[][]{
                {49.36111111111111, -13.555555555555554, 2.111111111111111},
                {-13.555555555555555, 3.7777777777777777, -0.5555555555555556},
                {2.111111111111111, -0.5555555555555556, 0.1111111111111111}};
        exp = new Matrix(expEntries);

        assertEquals(exp, Invert.invSymPosDef(A));

        // --------------------- Sub-case 3 ---------------------
        entries = new double[][]{
                {4, 12, -16},
                {12, 37, -43},
                {-16, -43, 98}};
        A = new Matrix(entries);
        expEntries = new double[][]{
                {49.36111111111111, -13.555555555555554, 2.111111111111111},
                {-13.555555555555555, 3.7777777777777777, -0.5555555555555556},
                {2.111111111111111, -0.5555555555555556, 0.1111111111111111}};
        exp = new Matrix(expEntries);

        assertEquals(exp, Invert.invSymPosDef(A));

        // --------------------- Sub-case 4 ---------------------
        entries = new double[][]{
                {4, 12, -16},
                {12, 2, -43},
                {-16, -43, 98}};
        A = new Matrix(entries);

        assertThrows(LinearAlgebraException.class, ()-> Invert.invSymPosDef(A, true));

        // --------------------- Sub-case 5 ---------------------
        entries = new double[][]{
                {4, 12, -16},
                {12, 37, -43},
                {1, -43, 98}};
        A = new Matrix(entries);

        assertThrows(LinearAlgebraException.class, ()-> Invert.invSymPosDef(A, true));
    }
}
