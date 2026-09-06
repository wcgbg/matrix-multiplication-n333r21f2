#include "core/gf.h"

#include <algorithm>

#include "gtest/gtest.h"

namespace {

TEST(GF2Test, AddSubXor) {
  using F = GF<2>;
  EXPECT_EQ(F::Add(F{0}, F{0}), F{0});
  EXPECT_EQ(F::Add(F{1}, F{0}), F{1});
  EXPECT_EQ(F::Add(F{1}, F{1}), F{0});
  EXPECT_EQ(F::Sub(F{0}, F{1}), F{1});
  EXPECT_EQ(F::Sub(F{1}, F{1}), F{0});
}

TEST(GF2Test, MulNeg) {
  using F = GF<2>;
  EXPECT_EQ(F::Mul(F{0}, F{0}), F{0});
  EXPECT_EQ(F::Mul(F{1}, F{1}), F{1});
  EXPECT_EQ(F::Mul(F{1}, F{0}), F{0});
  EXPECT_EQ(F::Neg(F{0}), F{0});
  EXPECT_EQ(F::Neg(F{1}), F{1});
}

TEST(GF2Test, Inverse) {
  using F = GF<2>;
  EXPECT_EQ(F::Inverse(F{1}), F{1});
}

TEST(GF3Test, AddSub) {
  using F = GF<3>;
  EXPECT_EQ(F::Add(F{1}, F{2}), F{0});
  EXPECT_EQ(F::Add(F{2}, F{2}), F{1});
  EXPECT_EQ(F::Sub(F{0}, F{1}), F{2});
  EXPECT_EQ(F::Sub(F{1}, F{2}), F{2});
}

TEST(GF3Test, MulNeg) {
  using F = GF<3>;
  EXPECT_EQ(F::Mul(F{2}, F{2}), F{1});
  EXPECT_EQ(F::Mul(F{2}, F{1}), F{2});
  EXPECT_EQ(F::Neg(F{1}), F{2});
  EXPECT_EQ(F::Neg(F{2}), F{1});
  EXPECT_EQ(F::Neg(F{0}), F{0});
}

TEST(GF3Test, Inverse) {
  using F = GF<3>;
  EXPECT_EQ(F::Inverse(F{1}), F{1});
  EXPECT_EQ(F::Inverse(F{2}), F{2}); // 2*2 = 4 = 1 mod 3
  for (uint8_t a = 1; a < 3; ++a) {
    EXPECT_EQ(F::Mul(F{a}, F::Inverse(F{a})), F::One()) << "a=" << int{a};
  }
}

TEST(GF5Test, Inverse) {
  using F = GF<5>;
  EXPECT_EQ(F::Inverse(F{1}), F{1});
  EXPECT_EQ(F::Inverse(F{2}), F{3}); // 2*3 = 6 = 1 mod 5
  EXPECT_EQ(F::Inverse(F{3}), F{2});
  EXPECT_EQ(F::Inverse(F{4}), F{4}); // 4*4 = 16 = 1 mod 5
  for (uint8_t a = 1; a < 5; ++a) {
    EXPECT_EQ(F::Mul(F{a}, F::Inverse(F{a})), F::One()) << "a=" << int{a};
  }
}

TEST(GF7Test, Inverse) {
  using F = GF<7>;
  for (uint8_t a = 1; a < 7; ++a) {
    EXPECT_EQ(F::Mul(F{a}, F::Inverse(F{a})), F::One()) << "a=" << int{a};
  }
}

TEST(GFConstexprTest, AllOpsAreConstexpr) {
  // Compile-time evaluation pins the constexpr-ness of every op.
  static_assert(GF<3>::Add(1, 2) == GF<3>::Zero());
  static_assert(GF<3>::Sub(1, 2) == GF<3>{2});
  static_assert(GF<3>::Mul(2, 2) == GF<3>::One());
  static_assert(GF<3>::Neg(1) == GF<3>{2});
  static_assert(GF<3>::Inverse(2) == GF<3>{2});
  static_assert(GF<5>::Inverse(2) == GF<5>{3});
  SUCCEED();
}

// --- exhaustive field-axiom checks -----------------------------------------

// Verify field axioms over F_P by exhaustive check.
template <int P> void CheckFieldAxioms() {
  using F = GF<P>;
  constexpr int q = F::kQ;
  // Additive identity & inverse.
  for (int a = 0; a < q; ++a) {
    F fa{static_cast<uint8_t>(a)};
    EXPECT_EQ(F::Add(fa, F::Zero()), fa) << "a=" << a;
    EXPECT_EQ(F::Add(fa, F::Neg(fa)), F::Zero()) << "a=" << a;
    EXPECT_EQ(F::Sub(fa, fa), F::Zero()) << "a=" << a;
  }
  // Commutativity of Add.
  for (int a = 0; a < q; ++a) {
    for (int b = 0; b < q; ++b) {
      F fa{static_cast<uint8_t>(a)};
      F fb{static_cast<uint8_t>(b)};
      EXPECT_EQ(F::Add(fa, fb), F::Add(fb, fa));
    }
  }
  // Multiplicative identity, inverses on nonzero, Inverse(0) = 0.
  EXPECT_EQ(F::Inverse(F::Zero()), F::Zero());
  for (int a = 1; a < q; ++a) {
    F fa{static_cast<uint8_t>(a)};
    EXPECT_EQ(F::Mul(fa, F::One()), fa) << "a=" << a;
    EXPECT_EQ(F::Mul(fa, F::Inverse(fa)), F::One()) << "a=" << a;
  }
  // Multiplicative zero.
  for (int a = 0; a < q; ++a) {
    F fa{static_cast<uint8_t>(a)};
    EXPECT_EQ(F::Mul(fa, F::Zero()), F::Zero()) << "a=" << a;
  }
  // Commutativity of Mul.
  for (int a = 0; a < q; ++a) {
    for (int b = 0; b < q; ++b) {
      F fa{static_cast<uint8_t>(a)};
      F fb{static_cast<uint8_t>(b)};
      EXPECT_EQ(F::Mul(fa, fb), F::Mul(fb, fa));
    }
  }
  // Distributivity across all triples.
  for (int a = 0; a < q; ++a) {
    for (int b = 0; b < q; ++b) {
      for (int c = 0; c < q; ++c) {
        F fa{static_cast<uint8_t>(a)};
        F fb{static_cast<uint8_t>(b)};
        F fc{static_cast<uint8_t>(c)};
        EXPECT_EQ(F::Mul(fa, F::Add(fb, fc)),
                  F::Add(F::Mul(fa, fb), F::Mul(fa, fc)))
            << "a=" << a << " b=" << b << " c=" << c;
      }
    }
  }
}

TEST(GF2Test, FieldAxioms) { CheckFieldAxioms<2>(); }
TEST(GF3Test, FieldAxioms) { CheckFieldAxioms<3>(); }
TEST(GF5Test, FieldAxioms) { CheckFieldAxioms<5>(); }
TEST(GF7Test, FieldAxioms) { CheckFieldAxioms<7>(); }

// Pow: identity at exp = 0; Fermat at exp = q − 1; matches iterated Mul for
// arbitrary small exponents.
template <int P> void CheckPow() {
  using F = GF<P>;
  constexpr int q = F::kQ;
  for (int a = 0; a < q; ++a) {
    F fa{static_cast<uint8_t>(a)};
    EXPECT_EQ(F::Pow(fa, 0), F::One()) << "a=" << a;
  }
  for (int a = 1; a < q; ++a) {
    F fa{static_cast<uint8_t>(a)};
    EXPECT_EQ(F::Pow(fa, q - 1), F::One()) << "a=" << a; // Fermat
  }
  // Cross-check against iterated Mul for small exponents.
  for (int a = 0; a < q; ++a) {
    F fa{static_cast<uint8_t>(a)};
    F acc = F::One();
    for (int e = 0; e < std::min(8, q + 2); ++e) {
      EXPECT_EQ(F::Pow(fa, e), acc) << "a=" << a << " e=" << e;
      acc = F::Mul(acc, fa);
    }
  }
}

TEST(GFPowTest, F2) { CheckPow<2>(); }
TEST(GFPowTest, F3) { CheckPow<3>(); }
TEST(GFPowTest, F5) { CheckPow<5>(); }
TEST(GFPowTest, F7) { CheckPow<7>(); }

// --- Operator overloads -----------------------------------------------------

TEST(GFOperatorsTest, F3ArithmeticOperators) {
  using F = GF<3>;
  const F a = 2, b = 1;
  EXPECT_EQ(a + b, F{0});
  EXPECT_EQ(a - b, F{1});
  EXPECT_EQ(a * b, F{2});
  EXPECT_EQ(-a, F{1});
  EXPECT_EQ(a.Inverse(), F{2});
  F c = a;
  c += b;
  EXPECT_EQ(c, F{0});
  c -= b;
  EXPECT_EQ(c, F{2});
  c *= a;
  EXPECT_EQ(c, F{1});
}

TEST(GFOperatorsTest, ImplicitFromUint8) {
  using F = GF<3>;
  F a = 2;
  F b = 1;
  EXPECT_EQ(a + b, F::Zero());
}

// --- ToString ---------------------------------------------------------------

TEST(GFToStringTest, F2) {
  using F = GF<2>;
  EXPECT_EQ(F{0}.ToString(), "0");
  EXPECT_EQ(F{1}.ToString(), "1");
}

TEST(GFToStringTest, F3) {
  using F = GF<3>;
  EXPECT_EQ(F{0}.ToString(), "0");
  EXPECT_EQ(F{1}.ToString(), "1");
  EXPECT_EQ(F{2}.ToString(), "2");
}

TEST(GFToStringTest, F13) {
  using F = GF<13>;
  EXPECT_EQ(F{0}.ToString(), "0");
  EXPECT_EQ(F{5}.ToString(), "5");
  EXPECT_EQ(F{9}.ToString(), "9");
  EXPECT_EQ(F{10}.ToString(), "10");
  EXPECT_EQ(F{12}.ToString(), "12");
}

TEST(GFOperatorsTest, Layout) {
  static_assert(sizeof(GF<2>) == 1);
  static_assert(sizeof(GF<3>) == 1);
  static_assert(std::is_trivially_copyable_v<GF<2>>);
  SUCCEED();
}

} // namespace
