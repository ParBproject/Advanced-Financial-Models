"""Print expected loss and concentration for the committed 1,000-loan book."""

from financial_models import (
    credit_book_benchmark,
    credit_book_memo_audit,
    credit_concentration,
    format_credit_book_benchmark,
    format_credit_book_decision,
    loans_from_credit_book,
    summarize_portfolio,
)

loans = loans_from_credit_book()
summary = summarize_portfolio(loans)
concentration = credit_concentration(loans)
audit = credit_book_memo_audit()

print(f"loans {len(summary.loans)}")
print(f"exposure {summary.total_exposure:.2f}")
print(f"expected_loss {summary.total_expected_loss:.2f}")
print(f"expected_loss_ratio {summary.expected_loss_ratio:.6f}")
print(f"hhi {concentration.herfindahl_hirschman_index:.8f}")
print(f"effective_borrowers {concentration.effective_borrower_count:.2f}")
print(format_credit_book_decision(summary, concentration, audit))
print(format_credit_book_benchmark(credit_book_benchmark()))
